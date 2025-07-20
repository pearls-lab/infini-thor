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


def generate(env, agent, data_save_path, scene_id, scene_id_to_objs, num_traj_per_scene=1, min_step=300, min_subgoal=10):
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

    succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])
    full_traj, fail_traj = set(), set()

    for traj_idx in range(num_existing, num_traj_per_scene):
        env.reset(scene_id)

        # setup data dictionary for a new traj to store. 
        # Planner stores each step automatically into constants.data_dict. See `DeterministicPlannerAgent`
        constants.data_dict = setup_data_dict()               

        traj_manager = TrajManager(scene_id=scene_id)
        
        # -- Main generation loop -- 
        # keeps trying out new task tuples as trajectories either fail or suceed
        max_fail_cnt = 0 # sometimes agent is stuck in the dead state; discard traj if max_fail_cnt > 20,
        while (traj_manager.n_steps < min_step or len(traj_manager.sub_traj_list) < min_subgoal) and max_fail_cnt <= 20:
            terminal, err = sample_and_simulate(env, agent, traj_manager, scene_id,
                                            succ_traj, full_traj, fail_traj,
                                            goal_candidates, pickup_candidates,
                                            movable_candidates, receptacle_candidates)
            
            if terminal and env.last_event.metadata['lastActionSuccess']:
                agent_holding = False
                if len(env.last_event.metadata['inventoryObjects']) > 0:
                    inv_obj = env.last_event.metadata['inventoryObjects'][0]['objectId']
                    print(f"## Agent is holding somthing: {inv_obj}, put it on the random location")
                    success, recep_obj = navigate_to(env, openable_objs, inv_obj=inv_obj, hasObjectToPut=True)
                    inv_obj_name = inv_obj.split("|")[0]
                    if success:
                        agent_holding = True
                        task_desc = f"put {inv_obj_name} on the {recep_obj}"
                        new_row = {"goal": "put object", "movable": inv_obj_name, "pickup": None,
                                    "receptacle": recep_obj, "scene": str(scene_id)}
                        constants.data_dict['plan']['high_pddl'].append({
                                "discrete_action": {
                                    "action": "PutObject",
                                    "args": [inv_obj_name, recep_obj]
                                },
                                "high_idx": 0,
                                "planner_action": {"action": "PutObject",
                                                    "objectId": inv_obj_name,
                                                    "receptacleObjectId": recep_obj
                                }
                        })
                        traj_manager.add_sub_traj(constants.data_dict['plan'])
                        traj_manager.add_sub_task(task_info=new_row, task_desc=task_desc, pddl_params=constants.data_dict['pddl_params'])
                        traj_manager.add_last_event(copy.deepcopy(env.last_event))
                    else:
                        traj_manager.discard_dead_end()
                        traj_manager.teleport_to_last_state(env)
                
                success, fail_at = traj_manager.replay_and_fix_objectIds(env)
                if success:
                    print("## Replay successed -- saving new traj")
                    max_fail_cnt = 0 # reset max fail cnt
                else:
                    traj_manager.discard_dead_end()
                    traj_manager.teleport_to_last_state(env)
                    if agent_holding: # do one more time for the additional put action
                        traj_manager.discard_dead_end()
                        traj_manager.teleport_to_last_state(env)
                    max_fail_cnt += 1
            else:
                max_fail_cnt += 1

            if max_fail_cnt > 20 and len(traj_manager.last_event_list) > 0:
                success = traj_manager.discard_dead_end()
                if success:
                    print("## Agent is stuck in a dead end -- Going back to the backup state.")
                    traj_manager.teleport_to_last_state(env)
                    max_fail_cnt = 0

            # reset constants.data_dict for next sub-traj
            constants.data_dict = setup_data_dict(scene_id=scene_id)

        if max_fail_cnt > 20:
            print("FAIL at generating. Ignore this traj")
            continue
            
        # replay and fix the objectIds
        success, fail_at = traj_manager.replay_and_fix_objectIds(env)

        if success and args.testset:
            success = generate_synthetic_task(env, agent, traj_manager, scene_id, succ_traj, full_traj, fail_traj, 
                                            goal_candidates, pickup_candidates, movable_candidates, receptacle_candidates)
        
        if success:
            # save the trajectory
            json_save_path = Path(floor_plan_path, f"floorplan{scene_id}_{len(traj_manager.sub_task_list)}_{traj_manager.n_steps}_{int(time.time())}.json")
            json_save_path.parent.mkdir(parents=True, exist_ok=True)
            traj_manager.save_traj(json_save_path)
            print("Done -- saved: ", traj_manager.n_steps, json_save_path)
            max_fail_cnt = 0
        else:
            print("### REPLAY FAILED")
            max_fail_cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="new_trajectories", help="where to save the generated data")
    parser.add_argument('--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id")
    parser.add_argument("--min_step", type=int, default=300)
    parser.add_argument("--min_subgoal", type=int, default=10)
    parser.add_argument("--testset", type=bool, default=False,
                        help="Generate test examples with synthetic tasks at the end of trajectories. "
                             "Runs an additional loop to create final synthetic tasks. "
                             "Use this flag to generate valid or test sets.")
    args = parser.parse_args()
    
    data_save_path = Path(args.save_path)
    data_save_path.parent.mkdir(parents=True, exist_ok=True)
 
    scene_id_to_objs = store_metadata()

    # create env and agent
    env = ThorEnv(x_display=args.x_display)

    game_state = TaskGameStateFullKnowledge(env)
    agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)

    goal_candidates = constants.GOALS[:]

    num_traj_per_scene = 1  # generate 1 trajs in each floor plan
    constants.RECORD_VIDEO_IMAGES = False
    
    for scene_type, ids in constants.SCENE_TYPE.items():
        id_list = list(ids)
        random.shuffle(id_list)
        for scene_id in id_list:
            generate(env, agent, data_save_path, scene_id, scene_id_to_objs,
                        num_traj_per_scene, args.min_step, args.min_subgoal)