from torchtitan.logging import logger

def visible(obj_list):
    return [obj for obj in obj_list if obj['visible']]

def find_visible_and_property(objects, objname, property_name):
    for obj in objects:
        if objname.lower() in obj['name'].lower() and obj['visible'] and obj[property_name]:
            print(obj['name'], obj['objectId'])
            return obj['objectId']
    return None

def visible_and_pickupable(objects, objname):
    return find_visible_and_property(objects, objname, 'pickupable')

def visible_and_receptacle(objects, objname):
    return find_visible_and_property(objects, objname, 'receptacle')

def visible_and_openable(objects, objname):
    return find_visible_and_property(objects, objname, 'openable')

def visible_and_sliceable(objects, objname):
    return find_visible_and_property(objects, objname, 'sliceable')

def visible_and_toggleable(objects, objname):
    return find_visible_and_property(objects, objname, 'toggleable')

def visible_and_isOpen(objects, objname): # closeable
    return find_visible_and_property(objects, objname, 'isOpen')


def post_processing_action(action, objects, objname=None):
    actions_map = {
        "OpenObject": visible_and_openable,
        "CloseObject": visible_and_isOpen,
        "PickupObject": visible_and_pickupable,
        "PutObject": visible_and_receptacle,
        "ToggleObjectOn": visible_and_toggleable,
        "ToggleObjectOff": visible_and_toggleable,
        "SliceObject": visible_and_sliceable,
    }
    
    try: 
        if action.startswith('PutObject'):
            receptacle_obj = action.split()[-1].strip()
            return "PutObject", visible_and_receptacle(objects, receptacle_obj)
        elif action.startswith('ToggleObject'):
            objname = action.split()[-1].strip()
            return "ToggleObject", visible_and_toggleable(objects, objname)

        for action_prefix, func in actions_map.items():
            if action.startswith('PutObject'):
                receptacle_obj = action.split()[-1].strip()
                return action_prefix, func(objects, receptacle_obj)
            elif action.startswith(action_prefix):
                objname = action.split(action_prefix)[-1].strip()
                return action_prefix, func(objects, objname)
    except: # action parsing error
        return None, None
    
    return None, None


def get_templated_high_pddl_desc(high_pddl):
    a_type = high_pddl['discrete_action']['action']
    args = high_pddl['discrete_action']['args'] if 'args' in high_pddl['discrete_action'] else None

    if 'objectId' in high_pddl['planner_action']:
        objectId = high_pddl['planner_action']['objectId']
        obj_name = objectId.split("|")[0]
    elif a_type == 'PutObject':
        obj_name = high_pddl['discrete_action']['args'][0]
        recep_name = high_pddl['discrete_action']['args'][1]
    
    if 'receptacleObjectId' in high_pddl['planner_action']:
        receptacleObjectId = high_pddl['planner_action']['receptacleObjectId']
        recep_name = receptacleObjectId.split("|")[0]

    templated_str = ""

    if 'GotoLocation' in a_type:
        templated_str = f"go to the {args[0]}"
    elif 'OpenObject' in a_type:
        templated_str = f"open the {obj_name}"
    elif 'CloseObject' in a_type:
        templated_str = f"close the {obj_name}"
    elif 'PickupObject' in a_type:
        templated_str = f"pick up the {obj_name}"
    elif 'PutObject' in a_type:
        templated_str = f"put the {obj_name} in the {recep_name}"
    elif 'CleanObject' in a_type:
        templated_str = f"wash the {obj_name}"
    elif 'HeatObject' in a_type:
        templated_str = f"heat the {obj_name}"
    elif 'CoolObject' in a_type:
        templated_str = f"cool the {obj_name}"
    elif 'ToggleObject' in a_type:
        templated_str = f"toggle {obj_name}"
    elif 'SliceObject' in a_type:
        templated_str = f"slice the {obj_name}"
    elif 'End' in a_type:
        templated_str = "<<STOP>>"

    return templated_str


ACT_TEMPLATE = {
    "RotateLeft": "RotateLeft",
    "RotateRight": "RotateRight",
    "MoveAhead": "MoveAhead",
    "LookUp": "LookUp",
    "LookDown": "LookDown",
    "OpenObject": "OpenObject [object]",
    "CloseObject": "CloseObject [object]",
    "PickupObject": "PickupObject [object]",
    "PutObject": "PutObject [object] [receptacle]",
    "ToggleObjectOn": "ToggleObjectOn [object]",
    "ToggleObjectOff": "ToggleObjectOff [object]",
    "SliceObject": "SliceObject [object]",
    "NoOp": "NoOp",}


def serialize_action(act):
    if act['action'].find("_") >= 0:
        act['action'] = act['action'].split("_")[0]
    
    template = ACT_TEMPLATE[act['action']]
    if 'objectId' in act:
        template = template.replace("[object]", act['objectId'].split("|")[0])
    if 'receptacleObjectId' in act:
        template = template.replace("[receptacle]", act['receptacleObjectId'].split("|")[0])
    return template

def setup_scene(env, traj, reward_type='dense'):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj['scene']['scene_num']
    object_poses = traj['scene']['object_poses']
    dirty_and_empty = traj['scene']['dirty_and_empty']
    object_toggles = traj['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    logger.info(f"Reset env: {scene_name}")
    last_event = env.reset(scene_name)
    last_event = env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    last_event = env.step(dict(traj['scene']['init_action']))

    return last_event