# this code is built upon 
# https://github.com/askforalfred/alfred/blob/master/gen/scripts/generate_trajectories.py

import os
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import copy
from pathlib import Path

import gen.constants as constants
from gen.agents.planner_agent import DeterministicPlannerAgent
from env.thor_env import ThorEnv
from gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from gen.utils.gen_util import *


def generate_synthetic_task(env, agent, traj_manager, scene_id, succ_traj, full_traj, fail_traj,
                            goal_candidates, pickup_candidates, movable_candidates, receptacle_candidates):
    '''Generate a synthetic task at the end of the trajectory.'''
    
    # save when the object was visible
    early_20p = traj_manager.n_steps // 5

    early_pickupable_objs, early_recep_objs = set(), set()
    for obj_log in traj_manager.object_logs[:early_20p]:
        early_pickupable_objs.update(obj_log['pickupable'])
        early_recep_objs.update([x for x in obj_log['visible'] if x in receptacle_candidates])

    late_pickupable_objs, late_recep_objs = set(), set()
    for obj_log in traj_manager.object_logs[-early_20p:]:
        late_pickupable_objs.update(obj_log['pickupable'])
        late_recep_objs.update([x for x in obj_log['visible'] if x in receptacle_candidates])
    
    succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])
    full_traj, fail_traj = set(), set()

    current_n_step = traj_manager.n_steps

    terminal = False
    while True:
        print("## *** Sample long-horizon task ***")
        if np.random.random() > 0.5:
            pickup_candidates = list(early_pickupable_objs)
            _recep_candis = list(late_recep_objs)
        else:
            pickup_candidates = list(late_pickupable_objs)
            _recep_candis = list(early_recep_objs)

        terminal, err = sample_and_simulate(env, agent, traj_manager, scene_id,
                                    succ_traj, fail_traj, full_traj,
                                    goal_candidates, pickup_candidates,
                                    movable_candidates, _recep_candis)

        if not terminal and err == "OUT_OF_CASE":
            print("### Sample is out of cases. This trajectory is discarded")
            break
        
        if terminal and env.last_event.metadata['lastActionSuccess']:
            success_lh, fail_at = traj_manager.revalidate_traj(env)
            if success_lh:
                traj_manager.long_horizon_task = {
                    "task_info": traj_manager.sub_task_list[-1]['task_info'],
                    "task_desc": traj_manager.sub_task_list[-1]['task_desc'],
                    "pddl_params": traj_manager.sub_task_list[-1]['pddl_params'],
                }
                return success_lh
            else:
                traj_manager.discard_dead_end()
                traj_manager.teleport_to_last_state(env)
                print("### Success generating a long-horizon task, but REPLAY FAILED -> retrying...")

            if fail_at < current_n_step:
                print("### exception: cannot replay ealier trajs (which were okay before...)")
                return False
        
    return False



def write_status(data_save_path, scene_id, traj_idx, traj_manager, max_fail_cnt, sample_fail_cnt, phase):
    """Atomic one-line JSON status for external monitoring."""
    status = {
        'scene': scene_id, 'attempt': traj_idx, 'phase': phase,
        'n_steps': traj_manager.n_steps, 'n_subgoals': len(traj_manager.sub_traj_list),
        'exec_fails': max_fail_cnt, 'sample_fails': sample_fail_cnt, 'ts': int(time.time()),
    }
    tmp = Path(data_save_path, '.status.tmp')
    with open(tmp, 'w') as f:
        json.dump(status, f)
    os.replace(str(tmp), str(Path(data_save_path, 'status.json')))


def try_bank(env, traj_manager, floor, max_tries=6):
    """Roll back until the trajectory replays cleanly, or it shrinks below `floor`.

    Used both on give-up and when the final validation fails: instead of losing a
    long episode outright, keep the longest prefix that reproduces. Replays are
    slightly stochastic (scene restore jitters object poses), so a failing
    configuration is retried once before anything is cut; a repeat failure is
    then truncated AT the failing subgoal -- dropping trailing subgoals cannot
    fix a failure in an early one.
    Returns True if the (possibly shortened) trajectory replayed successfully and
    still has at least `floor` steps.
    """
    last_fail = None
    for _ in range(max_tries):
        if traj_manager.n_steps < floor or len(traj_manager.sub_traj_list) == 0:
            return False
        success, fail_at = traj_manager.replay_and_fix_objectIds(env)
        if success:
            return traj_manager.n_steps >= floor
        fail_st = getattr(traj_manager, 'last_fail_subtraj', len(traj_manager.sub_traj_list) - 1)
        print(f"## bank: replay failed in subgoal {fail_st} (local step {fail_at})")
        if last_fail != (fail_st, fail_at):
            last_fail = (fail_st, fail_at)   # first failure here: retry, may be jitter
            continue
        # deterministic failure: keep only the prefix strictly before the failing subgoal
        keep = max(0, fail_st)
        while len(traj_manager.sub_traj_list) > keep:
            if not traj_manager.discard_dead_end():
                return False
        print(f"## bank: truncated to {traj_manager.n_steps} steps "
              f"({len(traj_manager.sub_traj_list)} subgoals)")
        last_fail = None
    return False


def save_traj_file(traj_manager, floor_plan_path, scene_id, gen_info):
    json_save_path = Path(floor_plan_path,
                          f"floorplan{scene_id}_{len(traj_manager.sub_task_list)}_{traj_manager.n_steps}_{int(time.time())}.json")
    json_save_path.parent.mkdir(parents=True, exist_ok=True)
    traj_manager.save_traj(json_save_path, gen_info=gen_info)
    print("Done -- saved: ", traj_manager.n_steps, json_save_path)
    return json_save_path


def generate(env_holder, data_save_path, scene_id, scene_id_to_objs, num_traj_per_scene=1, min_step=300, min_subgoal=10,
             max_fail=20, save_floor=0, replay_every=1, gen_info=None):
    env, agent = env_holder['env'], env_holder['agent']
    floor_plan_path = Path(data_save_path, f"floorplan{scene_id}")
    floor_plan_path.parent.mkdir(parents=True, exist_ok=True)

    openable_objs = get_openable_objs(scene_id)

    file_list = [f for f in floor_plan_path.glob('**/*') if f.is_file() and f.suffix == '.json']
    num_existing = len(file_list)

    # check if this floor plan already has enough trajs
    print(f"############ Generating synthetic episodes: Scene {scene_id}")
    print(f"############ {num_existing} trajs exists ====> generating {num_traj_per_scene-num_existing} trajs ")

    scene_objects = scene_id_to_objs[str(scene_id)]
    pickup_candidates, movable_candidates, receptacle_candidates = get_obj_candidates(scene_objects)

    for traj_idx in range(num_existing, num_traj_per_scene):
        # Stamp status immediately: a relaunched worker otherwise inherits the
        # previous process's status file, and an external watchdog reading it
        # would kill this healthy process before it finishes THOR startup.
        write_status(data_save_path, scene_id, traj_idx, TrajManager(scene_id=scene_id), 0, 0, 'starting')

        # fresh sampling pools per attempt: these accumulate every used/failed task
        # tuple, and carrying them across attempts eventually exhausts the pool so
        # every later sample fails instantly (observed after ~10h of farm running).
        succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])
        full_traj, fail_traj = set(), set()

        env, agent = env_holder['env'], env_holder['agent']
        env.reset(scene_id)

        # setup data dictionary for a new traj to store. 
        # Planner stores each step automatically into constants.data_dict. See `DeterministicPlannerAgent`
        constants.data_dict = setup_data_dict()               

        traj_manager = TrajManager(scene_id=scene_id)
        
        # -- Main generation loop -- 
        # `max_fail` counts failures of *executed* plans (env state may be damaged);
        # cheap failures -- unsatisfiable sample, planner found no plan -- retry
        # without consuming the budget, under a separate generous cap.
        max_fail_cnt = 0
        sample_fail_cnt = 0
        max_sample_fail = max(200, 10 * max_fail)
        consecutive_rollbacks = 0
        subgoals_since_replay = 0
        while (traj_manager.n_steps < min_step or len(traj_manager.sub_traj_list) < min_subgoal) \
                and max_fail_cnt <= max_fail and sample_fail_cnt <= max_sample_fail:
            try:
                terminal, err = sample_and_simulate(env, agent, traj_manager, scene_id,
                                                succ_traj, full_traj, fail_traj,
                                                goal_candidates, pickup_candidates,
                                                movable_candidates, receptacle_candidates)
            except Exception as e:
                # most likely the Unity process died; restart the controller and
                # resume from the last accepted state instead of losing the run
                print(f"## ENV CRASH during sampling: {e} -- restarting AI2THOR")
                env_holder['restart']()
                env, agent = env_holder['env'], env_holder['agent']
                if len(traj_manager.last_event_list) > 0:
                    traj_manager.teleport_to_last_state(env)
                max_fail_cnt += 1
                constants.data_dict = setup_data_dict(scene_id=scene_id)
                continue

            if err == "OUT_OF_CASE":
                print("### task tuple pool exhausted for this attempt")
                break

            write_status(data_save_path, scene_id, traj_idx, traj_manager,
                         max_fail_cnt, sample_fail_cnt, 'sampling')

            if terminal and env.last_event.metadata['lastActionSuccess']:
                # sample_and_simulate has already registered the accepted subgoal.
                # Disposing of a still-held object (e.g. the knife left in hand by
                # SliceObject) is appended as its OWN subgoal, with real navigation
                # actions so it replays like any other.
                subgoals_added = 1

                if len(env.last_event.metadata['inventoryObjects']) > 0:
                    inv_obj = env.last_event.metadata['inventoryObjects'][0]['objectId']
                    inv_obj_name = inv_obj.split("|")[0]
                    print(f"## Agent still holding {inv_obj} -- putting it down as its own subgoal")
                    constants.data_dict['plan'] = {'high_pddl': [], 'low_actions': [], 'desc': []}
                    put_ok, recep_obj, _ = put_down_held_object(
                        env, agent, traj_manager, openable_objs, scene_id)
                    if put_ok and constants.data_dict['plan']['low_actions']:
                        traj_manager.add_sub_traj(constants.data_dict['plan'])
                        traj_manager.add_sub_task(
                            task_info={"goal": "put object", "movable": inv_obj_name,
                                       "pickup": None, "receptacle": recep_obj,
                                       "scene": str(scene_id)},
                            task_desc=f"put the {inv_obj_name} on the {recep_obj}",
                            pddl_params=constants.data_dict['pddl_params'])
                        traj_manager.add_last_event(copy.deepcopy(env.last_event))
                        subgoals_added += 1
                    else:
                        # cannot free the hand: this branch is a dead end because
                        # every later subgoal would be planned with a full hand
                        print("## put-down failed -- discarding this subgoal")
                        traj_manager.discard_dead_end()
                        traj_manager.teleport_to_last_state(env)
                        subgoals_added = 0
                        max_fail_cnt += 1
                        constants.data_dict = setup_data_dict(scene_id=scene_id)
                        continue
                
                # full validation replay every `replay_every` accepted subgoals
                # (every subgoal replays the whole episode from step 0, which makes
                # total cost quadratic -- a coarser cadence keeps the same final
                # guarantee since the episode is always fully replayed before saving)
                subgoals_since_replay += 1
                fail_st = -1
                if subgoals_since_replay >= replay_every:
                    success, fail_at = traj_manager.replay_and_fix_objectIds(env)
                    if not success:
                        # scene restore is slightly stochastic -- retry once
                        # before cutting anything
                        fail_st = getattr(traj_manager, 'last_fail_subtraj', -1)
                        success, fail_at = traj_manager.replay_and_fix_objectIds(env)
                        fail_st = max(fail_st, getattr(traj_manager, 'last_fail_subtraj', -1))
                    subgoals_since_replay = 0
                    if success:
                        traj_manager.checkpoint()   # plan state that provably replays
                else:
                    success = True
                if success:
                    max_fail_cnt = 0
                    consecutive_rollbacks = 0
                else:
                    # truncate AT the failing subgoal: the failure is often far
                    # from the end, and growing past an unreproducible subgoal
                    # just builds work that can never be saved
                    keep = max(0, fail_st) if fail_st >= 0 else len(traj_manager.sub_traj_list) - 1
                    popped = 0
                    while len(traj_manager.sub_traj_list) > keep:
                        if not traj_manager.discard_dead_end():
                            break
                        popped += 1
                    print(f"## validation failed in subgoal {fail_st} -- "
                          f"truncated {popped} subgoal(s), back to {traj_manager.n_steps} steps")
                    traj_manager.teleport_to_last_state(env)
                    max_fail_cnt += 1
            elif traj_manager.last_sample_executed:
                max_fail_cnt += 1      # actions ran and failed: state may be damaged
            else:
                sample_fail_cnt += 1   # nothing executed: cheap, retry freely

            if max_fail_cnt > max_fail and len(traj_manager.last_event_list) > 0:
                # exponential rollback: repeated dead ends in the same neighborhood
                # mean the recent prefix is poisoned -- back out faster each time
                n_pop = min(2 ** consecutive_rollbacks, max(1, len(traj_manager.sub_traj_list) - 1))
                popped = 0
                for _ in range(n_pop):
                    if not traj_manager.discard_dead_end():
                        break
                    popped += 1
                if popped > 0:
                    print(f"## Dead end -- rolled back {popped} subgoal(s) "
                          f"(consecutive rollback #{consecutive_rollbacks + 1})")
                    traj_manager.teleport_to_last_state(env)
                    max_fail_cnt = 0
                    consecutive_rollbacks += 1
                    subgoals_since_replay = 0

            # reset constants.data_dict for next sub-traj
            constants.data_dict = setup_data_dict(scene_id=scene_id)

        gave_up = max_fail_cnt > max_fail or sample_fail_cnt > max_sample_fail
        if gave_up:
            # the trajectory did not reach min_step, but whatever validated prefix
            # remains is still a real long-horizon episode -- bank it if long enough
            if save_floor > 0 and traj_manager.n_steps >= save_floor:
                write_status(data_save_path, scene_id, traj_idx, traj_manager,
                             max_fail_cnt, sample_fail_cnt, 'banking')
                banked = try_bank(env, traj_manager, save_floor)
                if not banked and traj_manager.restore_checkpoint():
                    # current plan is poisoned by fix-up drift; bank the last
                    # state that passed a full validation replay instead
                    banked = try_bank(env, traj_manager, save_floor)
                if banked:
                    info = dict(gen_info or {}, banked_on_giveup=True)
                    save_traj_file(traj_manager, floor_plan_path, scene_id, info)
                    continue
            print("FAIL at generating. Ignore this traj")
            continue

        # final validation replay (always full, regardless of cadence)
        write_status(data_save_path, scene_id, traj_idx, traj_manager,
                     max_fail_cnt, sample_fail_cnt, 'final_replay')
        success, fail_at = traj_manager.replay_and_fix_objectIds(env)

        if not success:
            # do not throw the episode away: drop trailing subgoals until it
            # reproduces, and save if it is still worth keeping
            floor = save_floor if save_floor > 0 else min_step
            banked = try_bank(env, traj_manager, floor)
            if not banked and traj_manager.restore_checkpoint():
                banked = try_bank(env, traj_manager, floor)
            if banked:
                info = dict(gen_info or {}, banked_on_final_replay_fail=True)
                save_traj_file(traj_manager, floor_plan_path, scene_id, info)
            else:
                print("### REPLAY FAILED")
            continue

        if args.testset:
            success = generate_synthetic_task(env, agent, traj_manager, scene_id, succ_traj, full_traj, fail_traj, 
                                            goal_candidates, pickup_candidates, movable_candidates, receptacle_candidates)
            if not success:
                print("### synthetic task generation failed")
                continue

        save_traj_file(traj_manager, floor_plan_path, scene_id, gen_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="new_trajectories", help="where to save the generated data")
    parser.add_argument('--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id")
    parser.add_argument("--min_step", type=int, default=300)
    parser.add_argument("--min_subgoal", type=int, default=10)
    parser.add_argument("--scene_ids", type=str, default=None,
                        help="Comma-separated floor plan ids to generate for (e.g. '230,210'). "
                             "Default: sweep every scene in constants.SCENE_TYPE.")
    parser.add_argument("--num_traj_per_scene", type=int, default=1,
                        help="Number of trajectories to generate per floor plan.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (useful for parallel workers).")
    parser.add_argument("--filter_goals_by_scene", action="store_true",
                        help="(default behaviour; kept for compatibility)")
    parser.add_argument("--no_scene_goal_filter", action="store_true",
                        help="Sample all 7 goal types in every scene, including ones "
                             "constants.GOALS_VALID marks unachievable for the room type. "
                             "This was the original behaviour; it wastes most samples in "
                             "non-kitchen scenes.")
    parser.add_argument("--look_around", action="store_true",
                        help="Insert balanced LookUp/LookDown actions around navigation "
                             "stretches so the agent walks with its head up instead of "
                             "staring at the floor for the whole episode. Every interaction "
                             "still executes at its original horizon.")
    parser.add_argument("--save_floor", type=int, default=0,
                        help="If > 0: when an attempt gives up (or the final replay fails), "
                             "roll back to the longest prefix that replays cleanly and save it "
                             "if it has at least this many steps, instead of discarding hours "
                             "of validated work. 0 disables (original behaviour).")
    parser.add_argument("--replay_every", type=int, default=1,
                        help="Run the full validation replay every N accepted subgoals "
                             "(default 1 = after every subgoal, the original behaviour). "
                             "Replays cost O(episode length), so N>1 substantially speeds up "
                             "long-trajectory generation; the episode is always fully "
                             "replayed before saving regardless.")
    parser.add_argument("--max_fail", type=int, default=20,
                        help="Unproductive iterations tolerated before the last subgoal is rolled back. "
                             "Raise it (e.g. 60) when generating very long trajectories in scenes where "
                             "many sampled task tuples are invalid.")
    parser.add_argument("--testset", type=bool, default=False,
                        help="Generate test examples with synthetic tasks at the end of trajectories. "
                             "Runs an additional loop to create final synthetic tasks. "
                             "Use this flag to generate valid or test sets.")
    args = parser.parse_args()
    
    data_save_path = Path(args.save_path)
    data_save_path.parent.mkdir(parents=True, exist_ok=True)

    constants.INSERT_LOOK_AROUND = args.look_around

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    scene_id_to_objs = store_metadata()

    # env factory: lets generate() restart a crashed AI2THOR without losing the run
    def make_env_agent():
        env = ThorEnv(x_display=args.x_display)
        game_state = TaskGameStateFullKnowledge(env)
        agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)
        return env, agent

    env_holder = {}

    def restart_env():
        try:
            env_holder['env'].stop()
        except Exception:
            pass
        env_holder['env'], env_holder['agent'] = make_env_agent()

    env_holder['env'], env_holder['agent'] = make_env_agent()
    env_holder['restart'] = restart_env

    goal_candidates = constants.GOALS[:]

    def goals_for_scene(scene_id):
        '''Goal types constants.GOALS_VALID marks as achievable in this room type.'''
        if args.no_scene_goal_filter:
            return constants.GOALS[:]
        room = next((t for t, ids in constants.SCENE_TYPE.items() if scene_id in ids), None)
        if room is None:
            return constants.GOALS[:]
        return [g for g in constants.GOALS if room in constants.GOALS_VALID.get(g, set())]

    # provenance stamped into every saved trajectory
    def git_rev():
        try:
            import subprocess
            return subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    gen_info = {
        'min_step': args.min_step, 'min_subgoal': args.min_subgoal,
        'max_fail': args.max_fail, 'seed': args.seed,
        'save_floor': args.save_floor, 'replay_every': args.replay_every,
        'scene_goal_filter': not args.no_scene_goal_filter,
        'look_around': args.look_around,
        'git': git_rev(), 'created': int(time.time()),
    }

    num_traj_per_scene = args.num_traj_per_scene
    constants.RECORD_VIDEO_IMAGES = False

    if args.scene_ids:
        # generate only for the requested floor plans, in the order given
        scene_list = [int(s) for s in args.scene_ids.split(",") if s.strip()]
    else:
        scene_list = []
        for scene_type, ids in constants.SCENE_TYPE.items():
            id_list = list(ids)
            random.shuffle(id_list)
            scene_list.extend(id_list)

    for scene_id in scene_list:
        goal_candidates = goals_for_scene(scene_id)
        print(f"############ goal candidates for scene {scene_id}: {goal_candidates}")
        generate(env_holder, data_save_path, scene_id, scene_id_to_objs,
                    num_traj_per_scene, args.min_step, args.min_subgoal, args.max_fail,
                    save_floor=args.save_floor, replay_every=args.replay_every,
                    gen_info=dict(gen_info, scene=scene_id))