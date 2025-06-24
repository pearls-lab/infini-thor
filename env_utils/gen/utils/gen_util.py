# this code is built upon 
# https://github.com/askforalfred/alfred/blob/master/gen/scripts/generate_trajectories.py

import os
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import copy
from pathlib import Path

import gen.constants as constants
from gen.agents.planner_agent import DeterministicPlannerAgent
from gen.utils.game_util import object_id_to_name, get_objects_with_name_and_prop


# structures to help with constraint enforcement.
goal_to_required_variables = {"pick_and_place_simple": {"pickup", "receptacle"},
                              "pick_two_obj_and_place": {"pickup", "receptacle"},
                              "look_at_obj_in_light": {"pickup", "receptacle"},
                              "pick_clean_then_place_in_recep": {"pickup", "receptacle"},
                              "pick_heat_then_place_in_recep": {"pickup", "receptacle"},
                              "pick_cool_then_place_in_recep": {"pickup", "receptacle"},
                              "pick_and_place_with_movable_recep": {"pickup", "movable", "receptacle"}}
goal_to_pickup_type = {'pick_heat_then_place_in_recep': 'Heatable',
                       'pick_cool_then_place_in_recep': 'Coolable',
                       'pick_clean_then_place_in_recep': 'Cleanable'}
goal_to_receptacle_type = {'look_at_obj_in_light': "Toggleable"}
goal_to_invalid_receptacle = {'pick_heat_then_place_in_recep': {'Microwave'},
                              'pick_cool_then_place_in_recep': {'Fridge'},
                              'pick_clean_then_place_in_recep': {'SinkBasin'},
                              'pick_two_obj_and_place': {'CoffeeMachine', 'ToiletPaperHanger', 'HandTowelHolder'}}

scene_id_to_objs = {}


def setup_data_dict(scene_id=1):
    data_dict = OrderedDict()
    data_dict['scene'] = {'floor_plan': "", 'random_seed': -1, 'scene_num': scene_id, 'init_action': [],
                          'object_poses': [], 'dirty_and_empty': None, 'object_toggles': []}
    data_dict['pddl_params'] = {'object_target': -1, 'object_sliced': -1,
                                          'parent_target': -1, 'toggle_target': -1,
                                          'mrecep_target': -1}
    data_dict['pddl_state'] = []
    data_dict['plan'] = {'high_pddl': [], 'low_actions': [], 'desc': []}
    data_dict['images'] = []
    data_dict['sub_trajs'] = []
    data_dict['template'] = {'task_desc': "", 'high_descs': []}
    data_dict['long_horizon_task'] = {}

    return data_dict


def store_metadata():
    # Set up data structure to track dataset balance and use for selecting next parameters.
    # In actively gathering data, we will try to maximize entropy for each (e.g., uniform spread of goals,
    # uniform spread over patient objects, uniform recipient objects, and uniform scenes).

    # objects-to-scene and scene-to-objects database
    for scene_type, ids in constants.SCENE_TYPE.items():
        for id in ids:
            obj_json_file = os.path.join('gen', 'layouts', 'FloorPlan%d-objects.json' % id)
            with open(obj_json_file, 'r') as of:
                scene_objs = json.load(of)

            id_str = str(id)
            scene_id_to_objs[id_str] = scene_objs
    return scene_id_to_objs


def sample_task(succ_traj, full_traj, fail_traj,
                       goal_candidates, pickup_candidates, movable_candidates, receptacle_candidates,
                       inject_noise=10):
    # Get the current conditional distributions of all variables (goal/pickup/receptacle/scene).
    goal_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) + succ_traj.loc[
        (succ_traj['pickup'].isin(pickup_candidates) if 'pickup' in goal_to_required_variables[c] else True) &
        (succ_traj['movable'].isin(movable_candidates) if 'movable' in goal_to_required_variables[c] else True) &
        (succ_traj['receptacle'].isin(receptacle_candidates) if 'receptacle' in goal_to_required_variables[c] else True)]
            ['goal'].tolist().count(c)))  # Conditional.
                   * (1 / (1 + succ_traj['goal'].tolist().count(c)))  # Prior.
                   for c in goal_candidates]
    goal_probs = [w / sum(goal_weight) for w in goal_weight]

    pickup_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                           sum([succ_traj.loc[
                                    succ_traj['goal'].isin([g]) &
                                    (succ_traj['movable'].isin(movable_candidates)
                                     if 'movable' in goal_to_required_variables[g] else True) &
                                    (succ_traj['receptacle'].isin(receptacle_candidates)
                                     if 'receptacle' in goal_to_required_variables[g] else True)]
                                ['pickup'].tolist().count(c) for g in goal_candidates])))
                     * (1 / (1 + succ_traj['pickup'].tolist().count(c)))
                     for c in pickup_candidates]
    pickup_probs = [w / sum(pickup_weight) for w in pickup_weight]

    movable_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                            sum([succ_traj.loc[
                                     succ_traj['goal'].isin([g]) &
                                     (succ_traj['pickup'].isin(pickup_candidates)
                                      if 'pickup' in goal_to_required_variables[g] else True) &
                                     (succ_traj['receptacle'].isin(receptacle_candidates)
                                      if 'receptacle' in goal_to_required_variables[g] else True)]
                                 ['movable'].tolist().count(c) for g in goal_candidates])))
                      * (1 / (1 + succ_traj['movable'].tolist().count(c)))
                      for c in movable_candidates]
    movable_probs = [w / sum(movable_weight) for w in movable_weight]

    receptacle_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                               sum([succ_traj.loc[
                                        succ_traj['goal'].isin([g]) &
                                        (succ_traj['pickup'].isin(pickup_candidates)
                                         if 'pickup' in goal_to_required_variables[g] else True) &
                                        (succ_traj['movable'].isin(movable_candidates)
                                         if 'movable' in goal_to_required_variables[g] else True)]
                                    ['receptacle'].tolist().count(c) for g in goal_candidates])))
                         * (1 / (1 + succ_traj['receptacle'].tolist().count(c)))
                         for c in receptacle_candidates]
    receptacle_probs = [w / sum(receptacle_weight) for w in receptacle_weight]
    
    # Calculate the probability difference between each value and the maximum so we can iterate over them to find a
    # next-best candidate to sample subject to the constraints of knowing which will fail.
    diffs = [("goal", goal_candidates[idx], goal_probs[idx] - min(goal_probs))
             for idx in range(len(goal_candidates)) if len(goal_candidates) > 1]
    diffs.extend([("pickup", pickup_candidates[idx], pickup_probs[idx] - min(pickup_probs))
                  for idx in range(len(pickup_candidates)) if len(pickup_candidates) > 1])
    diffs.extend([("movable", movable_candidates[idx], movable_probs[idx] - min(movable_probs))
                  for idx in range(len(movable_candidates)) if len(movable_candidates) > 1])
    diffs.extend([("receptacle", receptacle_candidates[idx], receptacle_probs[idx] - min(receptacle_probs))
                  for idx in range(len(receptacle_candidates)) if len(receptacle_candidates) > 1])
    
    # Iteratively pop the next biggest difference until we find a combination that is valid (e.g., not already
    # flagged as impossible by the simulator).
    variable_value_by_diff = {}
    diffs_as_keys = []  # list of diffs; index into list will be used as key values.
    for _, _, diff in diffs:
        already_keyed = False
        for existing_diff in diffs_as_keys:
            if np.isclose(existing_diff, diff):
                already_keyed = True
                break
        if not already_keyed:
            diffs_as_keys.append(diff)

    for variable, value, diff in diffs:
        key = None
        for kidx in range(len(diffs_as_keys)):
            if np.isclose(diffs_as_keys[kidx], diff):
                key = kidx
        if key not in variable_value_by_diff:
            variable_value_by_diff[key] = []
        variable_value_by_diff[key].append((variable, value))
    
    for key, diff in sorted(enumerate(diffs_as_keys), key=lambda x: x[1], reverse=True):
        variable_value = variable_value_by_diff[key]
        random.shuffle(variable_value)
        for variable, value in variable_value: # ('goal', 'pick_and_place_simple'), ('goal', 'pick_two_obj_and_place')

            # Select a goal.
            if variable == "goal":
                gtype = value
                print("sampled goal '%s' with prob %.4f" % (gtype, goal_probs[goal_candidates.index(gtype)]))
                _goal_candidates = [gtype]

                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]

            # Select a pickup object.
            elif variable == "pickup":
                pickup_obj = value
                print("sampled pickup object '%s' with prob %.4f" %
                      (pickup_obj,  pickup_probs[pickup_candidates.index(pickup_obj)]))
                _pickup_candidates = [pickup_obj]

                _goal_candidates = goal_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]            

            # Select a movable object.
            elif variable == "movable":
                movable_obj = value
                print("sampled movable object '%s' with prob %.4f" %
                      (movable_obj,  movable_probs[movable_candidates.index(movable_obj)]))
                _movable_candidates = [movable_obj]
                _goal_candidates = [g for g in goal_candidates if g == 'pick_and_place_with_movable_recep']

                _pickup_candidates = pickup_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]      

            # Select a receptacle.
            elif variable == "receptacle":
                receptacle_obj = value
                print("sampled receptacle object '%s' with prob %.4f" %
                      (receptacle_obj, receptacle_probs[receptacle_candidates.index(receptacle_obj)]))
                _receptacle_candidates = [receptacle_obj]

                _goal_candidates = goal_candidates[:]
                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]

            # Select a scene.
            else:
                raise ValueError('Wrong variable')
            
            # Perform constraint propagation to determine whether this is a valid assignment.
            propagation_finished = False
            while not propagation_finished:
                assignment_lens = (len(_goal_candidates), len(_pickup_candidates), len(_movable_candidates),
                                   len(_receptacle_candidates))
                # Constraints on goal.
                _goal_candidates = [g for g in _goal_candidates if
                                    (g not in goal_to_pickup_type or
                                     len(set(_pickup_candidates).intersection(  # Pickup constraint.
                                        constants.VAL_ACTION_OBJECTS[goal_to_pickup_type[g]])) > 0)
                                    and (g not in goal_to_receptacle_type or
                                         np.any([r in constants.VAL_ACTION_OBJECTS[goal_to_receptacle_type[g]]
                                                for r in _receptacle_candidates]))  # Valid by goal receptacle const.
                                    and (g not in goal_to_invalid_receptacle or
                                         len(set(_receptacle_candidates).difference(
                                            goal_to_invalid_receptacle[g])) > 0)  # Invalid by goal receptacle const.
                                    ]

                # Define whether to consider constraints for each role based on current set of candidate goals.
                pickup_constrained = np.any(["pickup" in goal_to_required_variables[g] for g in _goal_candidates])
                movable_constrained = np.any(["movable" in goal_to_required_variables[g] for g in _goal_candidates])
                receptacle_constrained = np.any(["receptacle" in goal_to_required_variables[g]
                                                 for g in _goal_candidates])

                # Constraints on pickup obj.
                _pickup_candidates = [p for p in _pickup_candidates if
                                      np.any([g not in goal_to_pickup_type or
                                              p in constants.VAL_ACTION_OBJECTS[goal_to_pickup_type[g]]
                                              for g in _goal_candidates])  # Goal constraint.
                                      and (not movable_constrained or
                                           np.any([p in constants.VAL_RECEPTACLE_OBJECTS[m]
                                                  for m in _movable_candidates]))  # Movable constraint.
                                      and (not receptacle_constrained or
                                           np.any([r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                                  p in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                  for r in _receptacle_candidates]))  # Receptacle constraint.
                                      ]
                # Constraints on movable obj.
                _movable_candidates = [m for m in _movable_candidates if
                                       'pick_and_place_with_movable_recep' in _goal_candidates  # Goal constraint
                                       and (not pickup_constrained or
                                            np.any([p in constants.VAL_RECEPTACLE_OBJECTS[m]
                                                   for p in _pickup_candidates]))  # Pickup constraint.
                                       and (not receptacle_constrained or
                                            np.any([r in constants.VAL_RECEPTACLE_OBJECTS and
                                                   m in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                   for r in _receptacle_candidates]))  # Receptacle constraint.
                                       ]
                # Constraints on receptacle obj.
                _receptacle_candidates = [r for r in _receptacle_candidates if
                                          np.any([(g not in goal_to_receptacle_type or
                                                   r in constants.VAL_ACTION_OBJECTS[goal_to_receptacle_type[g]]) and
                                                  (g not in goal_to_invalid_receptacle or
                                                  r not in goal_to_invalid_receptacle[g])
                                                  for g in _goal_candidates])  # Goal constraint.
                                          and (not receptacle_constrained or
                                               r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                               np.any([p in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                       for p in _pickup_candidates]))  # Pickup constraint.
                                          and (not movable_constrained or
                                               r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                               np.any([m in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                       for m in _movable_candidates]))  # Movable constraint.
                                          ]

                if assignment_lens == (len(_goal_candidates), len(_pickup_candidates), len(_movable_candidates),
                                       len(_receptacle_candidates)):
                    propagation_finished = True

            candidate_lens = {"goal": len(_goal_candidates), "pickup": len(_pickup_candidates),
                              "movable": len(_movable_candidates), "receptacle": len(_receptacle_candidates)}

            if candidate_lens["goal"] == 0:
                # print("Goal over-constrained; skipping")
                continue
            
            if np.all([0 in [candidate_lens[v] for v in goal_to_required_variables[g]] for g in _goal_candidates]):
                continue

            # Ensure some combination of the remaining constraints is not in failures and is not already populated
            # by the target number of repeats.
            failure_ensured = True
            full_ensured = True
            for g in _goal_candidates:
                pickup_iter = _pickup_candidates if "pickup" in goal_to_required_variables[g] else ["None"]
                for p in pickup_iter:
                    movable_iter = _movable_candidates if "movable" in goal_to_required_variables[g] else ["None"]
                    for m in movable_iter:
                        receptacle_iter = _receptacle_candidates if "receptacle" in goal_to_required_variables[g] \
                            else ["None"]
                        for r in receptacle_iter:
                            if (g, p, m, r) not in fail_traj:
                                failure_ensured = False
                            if (g, p, m, r) not in full_traj:
                                full_ensured = False
                            if not failure_ensured and not full_ensured:
                                break
                        if not failure_ensured and not full_ensured:
                            break
                    if not failure_ensured and not full_ensured:
                        break
                if not failure_ensured and not full_ensured:
                    break
            if failure_ensured:
                continue
            if full_ensured:
                continue

            if candidate_lens["goal"] > 1 or np.any([np.any([candidate_lens[v] > 1
                                                             for v in goal_to_required_variables[g]])
                                                     for g in _goal_candidates]):
                sampled_task = sample_task(succ_traj, full_traj, fail_traj,
                                                  _goal_candidates, _pickup_candidates,
                                                  _movable_candidates, _receptacle_candidates)
                if sampled_task is None:
                    continue
            else:
                g = _goal_candidates[0]
                p = _pickup_candidates[0] if "pickup" in goal_to_required_variables[g] else "None"
                m = _movable_candidates[0] if "movable" in goal_to_required_variables[g] else "None"
                r = _receptacle_candidates[0] if "receptacle" in goal_to_required_variables[g] else "None"
                sampled_task = (g, p, m, r)

            return sampled_task

    return None # Discovered that there are no valid assignments remaining.


def get_obj_candidates(scene_objects):
    # Union objects that can be placed.
    pickup_candidates = list(set().union(
        *[constants.VAL_RECEPTACLE_OBJECTS[obj] for obj in constants.VAL_RECEPTACLE_OBJECTS]))
    pickup_candidates = [p for p in pickup_candidates if constants.OBJ_PARENTS[p] in scene_objects or p in scene_objects]
    
    #movable_candidates = list(set(constants.MOVABLE_RECEPTACLES).intersection(obj_to_scene_ids.keys()))
    movable_candidates = list(set(constants.MOVABLE_RECEPTACLES).intersection(set(scene_objects)))
    receptacle_candidates = [obj for obj in constants.VAL_RECEPTACLE_OBJECTS
                            if obj not in constants.MOVABLE_RECEPTACLES and obj in scene_objects] + \
                            [obj for obj in constants.VAL_ACTION_OBJECTS["Toggleable"]
                            if obj in scene_objects]

    # this is annoying.. Toaster is not a valid receptacle object
    if 'Toaster' in receptacle_candidates:
        receptacle_candidates.remove('Toaster')
    receptacle_candidates.sort()

    return pickup_candidates, movable_candidates, receptacle_candidates

def get_openable_objs(scene_id):
    obj_json_file = os.path.join('gen', 'layouts', 'FloorPlan%d-openable.json' % scene_id)
    with open(obj_json_file, 'r') as of:
        openable_objs = json.load(of)

    openable_objs = [(k.split("|")[0], v) for k, v in openable_objs.items()]
    return openable_objs


def get_constraint_obj(scene_id, gtype, pickup_obj, receptacle_obj, movable_obj, pickup_candidates, num_place_fails=0):
    '''
        This is an interal logic from the origianl generate_trajectory.py code from ALFRED
        Need to understand some internal constraints manually set by the ALFRED authors e.g., see the goal_to_invalid_receptacle
        I just used this constratins as it is.
    '''
    constraint_objs = {'repeat': [(constants.OBJ_PARENTS[pickup_obj],  # Generate multiple parent objs.
                                    np.random.randint(2 if gtype == "pick_two_obj_and_place" else 1, 
                                                        constants.PICKUP_REPEAT_MAX + 1))],
                        'sparse': [(receptacle_obj.replace('Basin', ''),
                                    num_place_fails * constants.RECEPTACLE_SPARSE_POINTS)]}
    if movable_obj != "None":
        constraint_objs['repeat'].append((movable_obj, np.random.randint(1, constants.PICKUP_REPEAT_MAX + 1)))

    for obj_type in scene_id_to_objs[str(scene_id)]:
        if (obj_type in pickup_candidates and
                obj_type != constants.OBJ_PARENTS[pickup_obj] and obj_type != movable_obj):
            constraint_objs['repeat'].append((obj_type,
                                            np.random.randint(1, constants.MAX_NUM_OF_OBJ_INSTANCES + 1)))
    if gtype in goal_to_invalid_receptacle:
        constraint_objs['empty'] = [(r.replace('Basin', ''), num_place_fails * constants.RECEPTACLE_EMPTY_POINTS)
                                    for r in goal_to_invalid_receptacle[gtype]]
    constraint_objs['seton'] = []
    if gtype == 'look_at_obj_in_light':
        constraint_objs['seton'].append((receptacle_obj, False))
    if num_place_fails > 0:
        print("Failed %d placements in the past; increased free point constraints: " % num_place_fails
            + str(constraint_objs))
    return constraint_objs


def sample_and_simulate(env, agent, traj_manager, scene_id,
                        succ_traj, full_traj, fail_traj,
                        goal_candidates, pickup_candidates,
                        movable_candidates, receptacle_candidates):
    sampled_task = sample_task(succ_traj, full_traj, fail_traj,
                                goal_candidates, pickup_candidates,
                                movable_candidates, receptacle_candidates)
    if sampled_task is None:
        return False, "OUT_OF_CASE"

    print(f"##  # of steps: {traj_manager.n_steps}, # of subtrajs: {len(traj_manager.sub_traj_list)}")
    gtype, pickup_obj, movable_obj, receptacle_obj = sampled_task
    print("## sampled tuple: " + str((gtype, pickup_obj, movable_obj, receptacle_obj)))

    # Agent reset to new scene. This is necessary for DeterministicPlannerAgent to use PDDL
    scene_info = {'scene_num': scene_id, 'random_seed': random.randint(0, 2 ** 32)}
    constraint_objs = get_constraint_obj(scene_id, gtype, pickup_obj, receptacle_obj, 
                                         movable_obj, pickup_candidates)
    info = agent.reset(scene=scene_info, objs=constraint_objs)

    if len(traj_manager.last_event_list) > 0: # agent.reset() relocates the agent, so we need to teleport to the previous state
        traj_manager.teleport_to_last_state(env)
        
    # Problem initialization with given constraints.
    task_objs = {'pickup': pickup_obj}
    if movable_obj != "None":
        task_objs['mrecep'] = movable_obj
    if gtype == "look_at_obj_in_light":
        task_objs['toggle'] = receptacle_obj
    else:
        task_objs['receptacle'] = receptacle_obj
    
    try:
        agent.setup_problem({'info': info}, scene=scene_info, objs=task_objs)
    except:
        return False, "FAIL at agent.setup_problem()"
    
    # Now that objects are in their initial places, record them.
    object_poses = [{'objectName': obj['name'].split('(Clone)')[0], 
                    'position': obj['position'],
                    'rotation': obj['rotation']}
                    for obj in env.last_event.metadata['objects'] if obj['pickupable']]
    dirty_and_empty = gtype == 'pick_clean_then_place_in_recep'
    object_toggles = [{'objectType': o, 'isOn': v} for o, v in constraint_objs['seton']]
    constants.data_dict['scene']['object_poses'] = object_poses
    constants.data_dict['scene']['dirty_and_empty'] = dirty_and_empty
    constants.data_dict['scene']['object_toggles'] = object_toggles
    
    if len(traj_manager.last_event_list) == 0: # inital object poses
        # Pre-restore the scene to cause objects to "jitter" like they will when the episode is replayed
        # based on stored object and toggle info. This should put objects closer to the final positions they'll
        # be inlay at inference time (e.g., mugs fallen and broken, knives fallen over, etc.).
        # event = env.step(dict(constants.data_dict['scene']['init_action']))
        print("Performing restore via thor_env API")
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        agent_init_action = {'action': 'TeleportFull',
                            'x': env.last_event.metadata['agent']['position']['x'],
                            'y': env.last_event.metadata['agent']['position']['y'],
                            'z': env.last_event.metadata['agent']['position']['z'],
                            'rotateOnTeleport': True,
                            'horizon': 30,
                            'rotation': env.last_event.metadata['agent']['rotation']['y']}

        traj_manager.init_action = agent_init_action
        traj_manager.add_last_event(copy.deepcopy(env.last_event))

    # plan & execute
    terminal = False
    action_dict = agent.get_action(None) # -> action_dict = {'action': 'Plan'}
    
    try:
        agent.step(action_dict) # agent.step({'action': 'Plan'}) -> this does whole plan
        _, terminal = agent.get_reward()
        
        action_dict = agent.get_action(None)
        agent.step(action_dict) # agent.step({'action': 'End'}) -> this terminate
        _, terminal = agent.get_reward()
    except Exception as e:
        print(f"## ERROR: {e}, Invalid Task: {env.last_event.metadata['errorMessage']}")

    if terminal and 'Target openable Receptacle is CLOSED' in env.last_event.metadata['errorMessage']:
        if constants.data_dict['plan']['low_actions'][-1]['api_action']['action'] == 'PutObject':
            put_action = constants.data_dict['plan']['low_actions'].pop(-1)
            recep_obj = put_action['api_action']['receptacleObjectId']

            env.step({'action': 'OpenObject', 'objectId': recep_obj})
            if env.last_event.metadata['lastActionSuccess']:               
                env.step(put_action['api_action'])
                if env.last_event.metadata['lastActionSuccess']:               
                    env.step({'action': 'CloseObject', 'objectId': recep_obj})
                    if env.last_event.metadata['lastActionSuccess']:
                        constants.data_dict['plan']['low_actions'].append({'high_idx': put_action['high_idx'], 'api_action': {'action': 'OpenObject', 'objectId': recep_obj}})
                        constants.data_dict['plan']['low_actions'].append(put_action)
                        constants.data_dict['plan']['low_actions'].append({'high_idx': put_action['high_idx'], 'api_action': {'action': 'CloseObject', 'objectId': recep_obj}})
            
    if terminal and env.last_event.metadata['lastActionSuccess']:
        # add to save structure.
        new_row = {"goal": gtype, "movable": movable_obj, "pickup": pickup_obj,
                    "receptacle": receptacle_obj, "scene": str(scene_id)}

        succ_traj = pd.concat([succ_traj, pd.DataFrame([new_row])], ignore_index=True)

        traj_manager.add_sub_traj(constants.data_dict['plan'])
        traj_manager.add_sub_task(task_info=new_row,
                                    task_desc=agent.game_state.get_task_str(),
                                    pddl_params=constants.data_dict['pddl_params'])
        traj_manager.add_last_event(copy.deepcopy(env.last_event))
    else:
        print(f"## Planner failed (terminal: {terminal}): {env.last_event.metadata['errorMessage']}")
        fail_traj.add((gtype, pickup_obj, movable_obj, receptacle_obj))
    
    # if this combination gave us the repeats we wanted, note it as filled.
    full_traj.add((gtype, pickup_obj, movable_obj, receptacle_obj))

    return terminal, env.last_event.metadata['errorMessage']


class TrajManager:

    def __init__(self, scene_id):
        self.scene_id = scene_id

        self.sub_traj_list = []
        self.sub_task_list = []
        self.last_event_list = []
        self.high_pddl = []
        self.low_actions = []
        self.init_action = None
        self.max_fail_cnt = 0
        self.n_steps = 0
        self.long_horizon_task = {}
        self.object_logs = []

    def add_last_event(self, event):
        self.last_event_list.append(event)

    def add_sub_task(self, task_info, task_desc, pddl_params):
        self.sub_task_list.append(dict(task_info=task_info, task_desc=task_desc, pddl_params=pddl_params))

    def add_sub_traj(self, sub_traj):
        self.sub_traj_list.append(sub_traj)
        self.n_steps += len(sub_traj['low_actions'])
        
    def discard_dead_end(self):
        if len(self.last_event_list) == 1: # preserve the init state
            return False
        self.last_event_list.pop(-1)
        self.sub_task_list.pop(-1)
        last_traj = self.sub_traj_list.pop(-1)
        self.n_steps -= len(last_traj['low_actions'])
        print("### discard the dead end -- # steps: ", self.n_steps)
        return True

    def save_traj(self, json_save_path):
        data_dict = setup_data_dict(self.scene_id)
        data_dict['scene']['init_action'] = self.init_action

        # Now that objects are in their initial places, record them.
        init_event = self.last_event_list[0]
        object_poses = [{'objectName': obj['name'].split('(Clone)')[0], 
                        'position': obj['position'],
                        'rotation': obj['rotation']}
                        for obj in init_event.metadata['objects'] if obj['pickupable']]
        data_dict['scene']['object_poses'] = object_poses
        
        # assign new index
        high_idx, low_idx = 0, 0
        assert len(self.sub_traj_list) == len(self.sub_task_list)
        for sub_traj_idx, subtraj in enumerate(self.sub_traj_list):
            start_high_idx = high_idx
            for hp in subtraj['high_pddl']:
                if hp['discrete_action']['action'] == "NoOp":
                    continue
                hp['high_idx'] = high_idx
                data_dict['plan']['high_pddl'].append(hp)
                high_idx +=1

            start_low_idx = low_idx
            for la in subtraj['low_actions']:
                la['low_idx'] = low_idx
                la['high_idx'] = start_high_idx + la['high_idx']
                data_dict['plan']['low_actions'].append(la)
                low_idx += 1
            
            data_dict['sub_trajs'].append({
                "high_pddl_idx": [start_high_idx, high_idx],
                "low_pddl_idx": [start_low_idx, low_idx],
                "sub_traj_idx": sub_traj_idx,
                "subgoal": self.sub_task_list[sub_traj_idx]['task_desc'],
            })

        data_dict['sub_tasks'] = self.sub_task_list
        data_dict['long_horizon_task'] = self.long_horizon_task
        data_dict['object_log'] = self.object_logs

        with open(json_save_path, "w") as fp:
            json.dump(data_dict, fp, sort_keys=True, indent=4)

        # push to HF
        filename = json_save_path.parts[-1]
        floorplan = json_save_path.parts[-2]

        s3_object_name = f"long_alfred_extra/{filename}"

        s3_client.upload_file(str(json_save_path), bucket_name, s3_object_name)
    
    def replay_and_fix_objectIds(self, env):
        env.reset(self.scene_id)
        env.step(self.init_action)

        init_event = self.last_event_list[0]
        object_poses = [{'objectName': obj['name'].split('(Clone)')[0], 
                        'position': obj['position'],
                        'rotation': obj['rotation']}
                        for obj in init_event.metadata['objects'] if obj['pickupable']]
        env.step((dict(action='SetObjectPoses', objectPoses=object_poses)))

        self.object_logs = []

        replay_success = True
        fail_at = -1
        for subtraj in self.sub_traj_list:
            for lidx, la in enumerate(subtraj['low_actions']):
                traj_api_cmd = la['api_action']
                traj_api_cmd['renderImage'] = True
                env.step(traj_api_cmd)
                if not env.last_event.metadata['lastActionSuccess']: # something went wrong.
                    if 'Object ID appears to be invalid' in env.last_event.metadata['errorMessage'] or \
                        'object not found' in env.last_event.metadata['errorMessage'] or \
                            'is not visible' in env.last_event.metadata['errorMessage']:
                        object_id = traj_api_cmd['objectId']
                        obj_name = object_id_to_name(object_id)
                        
                        if traj_api_cmd['action'] == "PickupObject":
                            # Pick the first one -- most cases are covered. exception is sliced thing
                            # This will pick random spliced object. e.g., 'Tomato|+00.32|+00.95|-02.41|TomatoSliced_4'
                            correct_obj = get_objects_with_name_and_prop(obj_name, 'pickupable', env.last_event.metadata)[0]  
                        elif traj_api_cmd['action'] == "SliceObject":
                            correct_obj = get_objects_with_name_and_prop(obj_name, 'sliceable', env.last_event.metadata)[0]    
                        elif traj_api_cmd['action'] == "PutObject":
                            correct_obj = env.last_event.metadata['inventoryObjects'][0]
                        elif "ToggleObject" in traj_api_cmd['action']:
                            correct_obj = get_objects_with_name_and_prop(obj_name, 'Toggleable', env.last_event.metadata)[0]
                        elif traj_api_cmd['action'] == "OpenObject":
                            correct_obj = get_objects_with_name_and_prop(obj_name, 'openable', env.last_event.metadata)[0]
                        elif traj_api_cmd['action'] == "CloseObject":
                            correct_obj = get_objects_with_name_and_prop(obj_name, 'isOpen', env.last_event.metadata)[0]
                        else:
                            return False, lidx

                        if object_id != correct_obj['objectId']:
                            la['api_action']['objectId'] = correct_obj['objectId']
                            print(f"### ({lidx}) Fixed Object Id: {object_id} ==> {correct_obj['objectId']}")
                            la['api_action']['renderImage'] = True
                            env.step(la['api_action'])
                            if not env.last_event.metadata['lastActionSuccess']:
                                print(f"### ({lidx}) this is not fixable: {env.last_event.metadata['errorMessage']}")           
                                replay_success = False
                                fail_at = lidx
                                return False, lidx
                    else:
                        print(lidx, traj_api_cmd)
                        print(f"### This error is not handled (at {lidx}): {env.last_event.metadata['errorMessage']}")
                        return False, lidx
                else:
                    objs = env.last_event.metadata['objects']
                    visibles = list(set([o['name'].split("_")[0] for o in objs if o['visible']]))
                    pickupable = list(set([o['name'].split("_")[0] for o in objs if o['pickupable']]))
                    isOpen = list(set([o['name'].split("_")[0] for o in objs if o['isOpen']]))

                    self.object_logs.append(dict(
                        t=lidx,
                        action=traj_api_cmd['action'],
                        visible=visibles,
                        pickupable=pickupable,
                        isOpen=isOpen
                    ))

        return replay_success, fail_at

    def replay(self, env):
        env.reset(self.scene_id)
        env.step(self.init_action)

        init_event = self.last_event_list[0]
        object_poses = [{'objectName': obj['name'].split('(Clone)')[0], 
                        'position': obj['position'],
                        'rotation': obj['rotation']}
                        for obj in init_event.metadata['objects'] if obj['pickupable']]
        env.step((dict(action='SetObjectPoses', objectPoses=object_poses)))

        replay_success = True
        fail_at = -1
        for subtraj in self.sub_traj_list:
            for lidx, la in enumerate(subtraj['low_actions']):
                traj_api_cmd = la['api_action']
                traj_api_cmd['renderImage'] = True
                env.step(traj_api_cmd)
                if not env.last_event.metadata['lastActionSuccess']:
                    return False, lidx
        return replay_success, fail_at        

    def revalidate_traj(self, env):
        '''
            Valid traj sometimes fails. Try three-times.
        '''
        for _ in range(3):
            success, fail_at = self.replay_and_fix_objectIds(env)
            if success:
                return self.replay(env)
                #return True, fail_at
        return False, fail_at

    def teleport_to_last_state(self, env):
        print("## Tellport to the previous state and restore object poses")
        pre_state_metadata = self.last_event_list[-1].metadata
        pre_objectposes = [{'objectName': obj['name'].split('(Clone)')[0],
                            'position': obj['position'],
                            'rotation': obj['rotation']}
                            for obj in pre_state_metadata['objects'] if obj['pickupable']]
        env.step({'action': 'TeleportFull',
                            'x': pre_state_metadata['agent']['position']['x'],
                            'y': pre_state_metadata['agent']['position']['y'],
                            'z': pre_state_metadata['agent']['position']['z'],
                            'rotation': pre_state_metadata['agent']['rotation']['y'],
                            'rotateOnTeleport': True,
                            'horizon': pre_state_metadata['agent']['cameraHorizon']})
        env.step((dict(action='SetObjectPoses', objectPoses=pre_objectposes)))


def navigate_to(env, openable_objs, inv_obj=None, hasObjectToPut=False):
    print("## Move to the other random location")
    pre_state_metadata = copy.deepcopy(env.last_event.metadata)

    random.shuffle(openable_objs)

    error_cnt = defaultdict(int)
    for (recep_obj, point) in openable_objs:
        try:
            env.step(dict(action='Teleport', 
                x=point[0],
                y=env.last_event.metadata['agent']['position']['y'],
                z=point[1],
                rotation=point[2],
                horizon=point[3]))

            if hasObjectToPut and env.last_event.metadata['lastActionSuccess']:
                candi_objs = [x for x in env.last_event.metadata['objects'] if x['visible'] and x['receptacle'] and x['name'].startswith(recep_obj)]
                env.step(dict(action='PutObject',
                                objectId=inv_obj,
                                receptacleObjectId=candi_objs[0]['objectId'],
                                forceAction=True,
                                placeStationary=True
                ))
                if env.last_event.metadata['lastActionSuccess']:
                    return True, recep_obj
            elif env.last_event.metadata['lastActionSuccess']:
                return True, recep_obj
            else:
                raise ValueError(" navigation Fail")
        except Exception as e:
            print(f"## !! Dead end !! : Fail to get to {recep_obj} -- Error: {e} ")
            teleport(env, pre_state_metadata)
            error_cnt[str(e)] += 1
            if error_cnt[str(e)] > 5:
                return False, None

    print("## !! DEAD END !!")
    return False, None


def teleport(env, metadata):
    env.step({'action': 'TeleportFull',
            'x': metadata['agent']['position']['x'],
            'y': metadata['agent']['position']['y'],
            'z': metadata['agent']['position']['z'],
            'rotation': metadata['agent']['rotation']['y'],
            'rotateOnTeleport': True,
            'horizon': metadata['agent']['cameraHorizon']})
