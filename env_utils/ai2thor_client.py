import requests
import json
import time
import argparse
from env_utils.env.tasks import get_task
import env_utils.gen.constants as constants

from torchtitan.logging import logger

#import logging
#logger = logging.getLogger()


class ThorEnv():
    '''
    Environment class that interacts with ai2thor.controller.Controller
    over HTTP. Includes extension codes for ALFRED tasks.
    adopted by: https://github.com/askforalfred/alfred/blob/master/env/thor_env.py
    '''
    def __init__(self, port=5000):
        self.task = None
        self.last_event = None
        self.scene_num = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # intemediate states for CoolObject Subgoal
        self.cooled_reward = False
        self.reopen_reward = False

        self.client = AI2THORClient(service_port=port)

        print("ThorEnv started.")

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num
        self.scene_num = scene_name_or_num.replace("FloorPlan", "")

        event = self.client.initialize(scene_name) # controller.reset
        
        event = self.step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibility_distance=visibility_distance,
            makeAgentsVisible=False,
        ))

        if self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        self.step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibility_distance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))

        if len(object_toggles) > 0:
            self.step((dict(action='SetObjectToggles', objectToggles=object_toggles)))

        if dirty_and_empty:
            self.step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeDirty",
                               forceAction=True))
            self.step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeFilled",
                               forceAction=False))

        last_event = self.step((dict(action='SetObjectPoses', objectPoses=object_poses)))
        return last_event

    
    #def set_task(self, traj, last_event, sub_traj_idx=None, task_info=None, task_type=None, num_subgoals=None, reward_type='sparse', max_episode_length=2000):
    def set_task(self, task_info, num_subgoals, last_event, expert_plan=None):
        '''
        set the current task type (one of 7 tasks)
        '''
        # if not task_type:
        #     if 'long_horizon_task' in traj:
        #         task_type = traj['long_horizon_task']['task_info']['goal']
        #     else:
        #         task_type = traj['taks_type']
        # self.task = get_task(task_type, traj,
        #             last_event,
        #             sub_traj_idx=sub_traj_idx,
        #             task_info=task_info,
        #             num_subgoals=num_subgoals,
        #             reward_type=reward_type,
        #             max_episode_length=max_episode_length)
        self.task = get_task(task_info, num_subgoals, last_event, expert_plan)

    def step(self, action: dict):
        last_event = self.client.interact(action)
        if last_event['success']:
            self.last_event = self.update_states(action, last_event)
            self.check_post_conditions(action)
        return self.last_event

    def restore_agent_last_state(self, expert):
        last_state = expert.last_event

        # if len(object_toggles) > 0:
        #     self.step((dict(action='SetObjectToggles', objectToggles=object_toggles)))

        # if dirty_and_empty:
        #     self.step(dict(action='SetStateOfAllObjects',
        #                        StateChange="CanBeDirty",
        #                        forceAction=True))
        #     self.step(dict(action='SetStateOfAllObjects',
        #                        StateChange="CanBeFilled",
        #                        forceAction=False))
        object_poses = [{'objectName': obj['name'], 
                        'position': obj['position'],
                        'rotation': obj['rotation']}
                        for obj in last_state['objects']]

        last_event = self.step((dict(action='SetObjectPoses', objectPoses=object_poses)))

        self.cleaned_objects = last_state['env_state']['cleaned_objects']
        self.cooled_objects = last_state['env_state']['cooled_objects']
        self.heated_objects = last_state['env_state']['heated_objects']

        last_event = self.step({'action': 'TeleportFull',
                    'x': last_state['agent']['position']['x'],
                    'y': last_state['agent']['position']['y'],
                    'z': last_state['agent']['position']['z'],
                    'rotation': last_state['agent']['rotation']['y'],
                    'rotateOnTeleport': True,
                    'horizon': last_state['agent']['cameraHorizon']})

        expert.last_state = last_event
        return last_event

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action, event):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        if event['lastActionSuccess']:
            event_metadata = {'objects': event['objects']} # make compatiable with util funcs below
            # clean
            if action == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = get_obj_of_type_closest_to_obj('SinkBasin', action['objectId'], event_metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = get_objects_of_type('Microwave', event_metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = get_objects_of_type('Fridge', event_metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        event['env_state'] = {
            "cleaned_objects": self.cleaned_objects,
            "cooled_objects": self.cooled_objects,
            "heated_objects": self.heated_objects,
        }

        return event

    def get_transition_reward(self, last_event, eval_idx=None, expert=False):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(last_event, self, eval_idx)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self, self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self, self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=inventory_object_id,
                          receptacleObjectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", object_id, event)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event)['isDirty']):
                    event = self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids


class AI2THORClient:
    def __init__(self, service_port=5000, service_url="http://localhost"):
        self.service_url = f"{service_url}:{service_port}"
        self.check_service()
    
    def check_service(self):
        """Check if the AI2THOR service is running"""
        try:
            response = requests.get(f"{self.service_url}/status")
            if response.status_code == 200:
                print("AI2THOR service is running")
            else:
                print("AI2THOR service returned unexpected response")
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot connect to AI2THOR service. Make sure it's running.")
            exit(1)
    
    def initialize(self, scene="FloorPlan1"):
        """Initialize the AI2THOR environment"""
        response = requests.post(
            f"{self.service_url}/initialize",
            json={"scene": scene}
        )
        return response.json()
    
    def interact(self, action):
        """Send an action to the AI2THOR environment"""
        response = requests.post(
            f"{self.service_url}/interact",
            json={"action": action}
        )
        return response.json()
