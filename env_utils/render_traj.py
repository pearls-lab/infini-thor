"""Replay a generated infini-THOR trajectory and dump RGB frames + a web-friendly timeline.

The generation script (`generate_traj.py`) stores trajectories as action plans only --
frames are not recorded, because rendering during search would make generation far slower.
This script deterministically replays a saved trajectory in AI2THOR and writes:

    out_dir/frames/%06d.jpg   one frame per low-level action (plus frame 0 = initial state)
    out_dir/timeline.json     per-step action / subgoal / object-observation metadata

`timeline.json` is what the project website consumes to drive the step counter, the
subgoal ticker and the "objects observed" HUD, so it is kept small and self-describing.

Example
-------
    python render_traj.py \
        --traj_json /gen_out/work/fp230_a/new_trajectories/floorplan230/floorplan230_58_2043_*.json \
        --out_dir /gen_out/render/fp230_a --width 640 --height 640
"""

import os
import math
import json
import time
import argparse
from pathlib import Path

import cv2

import gen.constants as constants
from env.thor_env import ThorEnv
from gen.utils.game_util import get_objects_with_name_and_prop, object_id_to_name



NAV_ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown")


def _pose(env):
    m = env.last_event.metadata["agent"]
    return dict(position=dict(m["position"]), rotation=m["rotation"]["y"],
                horizon=round(env.last_event.metadata["agent"]["cameraHorizon"], 4))


def _teleport(env, pos, rot, horizon):
    return env.step(dict(action="TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
                         rotation=round(rot, 3), horizon=round(horizon, 3), standing=True))


def _ease(a):
    """Smoothstep: ease in and out instead of a constant-velocity ramp.

    A linear sweep still reads as mechanical because the motion starts and stops
    abruptly at each action boundary; easing makes consecutive actions blend."""
    return a * a * (3.0 - 2.0 * a)


def tween_frames(env, api_cmd, n):
    """Render n-1 in-between frames for a navigation action, then rewind.

    AI2THOR executes MoveAhead/Rotate/Look as instantaneous 0.25 m / 90 deg / 30 deg
    jumps, which look like a strobe when replayed as video. We interpolate the pose
    with TeleportFull to get intermediate views, then teleport back so the real
    action still executes from the exact recorded state -- the replay itself stays
    bit-identical to the trajectory that was validated at generation time.

    Rotations get proportionally more in-between frames than translations: a 90
    degree turn sweeps far more of the view than a 0.25 m step, so at equal frame
    counts it looks much choppier.
    """
    if n <= 1 or api_cmd["action"] not in NAV_ACTIONS:
        return []
    if api_cmd["action"] in ("RotateLeft", "RotateRight"):
        n = n * 2          # 90 deg turn: twice the frames of a 0.25 m step

    start = _pose(env)
    act = api_cmd["action"]
    end_pos, end_rot, end_hor = dict(start["position"]), start["rotation"], start["horizon"]

    if act == "MoveAhead":
        mag = api_cmd.get("moveMagnitude", constants.AGENT_STEP_SIZE)
        rad = math.radians(start["rotation"])
        end_pos["x"] += mag * math.sin(rad)
        end_pos["z"] += mag * math.cos(rad)
    elif act in ("RotateLeft", "RotateRight"):
        end_rot = start["rotation"] + (-90 if act == "RotateLeft" else 90)
    else:  # LookUp / LookDown
        end_hor = start["horizon"] + constants.AGENT_HORIZON_ADJ * (-1 if act == "LookUp" else 1)

    frames = []
    for k in range(1, n):
        a = _ease(k / float(n))
        pos = {ax: start["position"][ax] * (1 - a) + end_pos[ax] * a for ax in "xyz"}
        ev = _teleport(env, pos, start["rotation"] * (1 - a) + end_rot * a,
                       start["horizon"] * (1 - a) + end_hor * a)
        if ev.metadata["lastActionSuccess"]:
            frames.append(ev.frame)

    # rewind so the recorded action runs from the true pre-action state
    _teleport(env, start["position"], start["rotation"], start["horizon"])
    return frames


def _obj_name(obj):
    return obj['name'].split('(Clone)')[0].split('_')[0]


def _fix_object_id(env, api_cmd):
    """Mirror the object-id repair TrajManager.replay_and_fix_objectIds does, so a
    trajectory that was validated at generation time also replays here."""
    if 'objectId' not in api_cmd:
        return None
    obj_name = object_id_to_name(api_cmd['objectId'])
    objs = env.last_event.metadata['objects']
    action = api_cmd['action']
    try:
        if action == 'PickupObject':
            return get_objects_with_name_and_prop(obj_name, 'pickupable', objs)[0]
        if action == 'SliceObject':
            return get_objects_with_name_and_prop(obj_name, 'sliceable', objs)[0]
        if action == 'PutObject':
            return env.last_event.metadata['inventoryObjects'][0]
        if 'ToggleObject' in action:
            return get_objects_with_name_and_prop(obj_name, 'Toggleable', objs)[0]
        if action == 'OpenObject':
            return get_objects_with_name_and_prop(obj_name, 'openable', objs)[0]
        if action == 'CloseObject':
            return get_objects_with_name_and_prop(obj_name, 'isOpen', objs)[0]
    except IndexError:
        return None
    return None


def render(traj_data, out_dir, args):
    out_dir = Path(out_dir)
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    scene_num = traj_data['scene']['scene_num']
    low_actions = traj_data['plan']['low_actions']

    # subgoal index -> text, and low_idx -> subgoal index
    sub_of_step = [0] * len(low_actions)
    subgoals = []
    for st in traj_data['sub_trajs']:
        lo, hi = st['low_pddl_idx']
        subgoals.append({'idx': st['sub_traj_idx'], 'text': st['subgoal'], 'start': lo, 'end': hi})
        for t in range(lo, min(hi, len(low_actions))):
            sub_of_step[t] = st['sub_traj_idx']

    env = ThorEnv(x_display=args.x_display,
                  player_screen_width=args.width,
                  player_screen_height=args.height,
                  quality=args.quality)
    env.reset(scene_num, render_image=True, render_depth_image=False,
              render_class_image=False, render_object_image=False)
    env.step(traj_data['scene']['init_action'])
    env.step(dict(action='SetObjectPoses', objectPoses=traj_data['scene']['object_poses']))

    def save(idx, frame):
        cv2.imwrite(str(frame_dir / ('%06d.jpg' % idx)),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    save(0, env.last_event.frame)
    n_frames = 1
    step_frames = []          # first frame index of each step, for the web player

    steps, first_seen = [], {}
    seen = set()
    t0 = time.time()
    for t, la in enumerate(low_actions):
        api_cmd = dict(la['api_action'])
        api_cmd['renderImage'] = True

        step_frames.append(n_frames)
        for f in tween_frames(env, api_cmd, args.smooth):
            save(n_frames, f)
            n_frames += 1

        env.step(api_cmd)

        if not env.last_event.metadata['lastActionSuccess']:
            correct = _fix_object_id(env, api_cmd)
            if correct is not None and correct['objectId'] != api_cmd.get('objectId'):
                api_cmd['objectId'] = correct['objectId']
                env.step(api_cmd)
            if not env.last_event.metadata['lastActionSuccess']:
                print('### replay diverged at step %d: %s' %
                      (t, env.last_event.metadata['errorMessage']))
                if not args.keep_going:
                    break

        save(n_frames, env.last_event.frame)
        n_frames += 1

        visible = sorted({_obj_name(o) for o in env.last_event.metadata['objects'] if o['visible']})
        for name in visible:
            if name not in first_seen:
                first_seen[name] = t
        seen.update(visible)

        steps.append({
            't': t,
            'a': api_cmd['action'],
            'o': object_id_to_name(api_cmd['objectId']) if 'objectId' in api_cmd else None,
            'sg': sub_of_step[t],
            'nseen': len(seen),
        })

        if (t + 1) % 100 == 0:
            rate = (t + 1) / (time.time() - t0)
            print('rendered %d/%d steps (%.1f steps/s)' % (t + 1, len(low_actions), rate))

    timeline = {
        'scene': 'FloorPlan%d' % scene_num,
        'source_json': os.path.basename(args.traj_json),
        'n_steps': len(steps),
        'n_frames': n_frames,
        'step_frames': step_frames[:len(steps)],
        'smooth': args.smooth,
        'n_subgoals': len(subgoals),
        'n_objects_seen': len(seen),
        'width': args.width,
        'height': args.height,
        'subgoals': subgoals,
        'long_horizon_task': traj_data.get('long_horizon_task', {}),
        'first_seen': first_seen,
        'steps': steps,
    }
    with open(out_dir / 'timeline.json', 'w') as f:
        json.dump(timeline, f)

    print('Done -- %d frames for %d steps, %d subgoals, %d distinct objects observed -> %s'
          % (n_frames, len(steps), len(subgoals), len(seen), out_dir))
    env.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_json', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--x_display', type=str, default=constants.X_DISPLAY)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--quality', type=str, default='MediumCloseFitShadows',
                        help="AI2THOR render quality, e.g. 'Ultra' for the nicest frames")
    parser.add_argument('--jpeg_quality', type=int, default=88)
    parser.add_argument('--smooth', type=int, default=1,
                        help='in-between frames per navigation action (1 = off). Interpolates the '
                             'agent pose with TeleportFull so replays are not a strobe of 0.25m / '
                             '90-degree jumps; the executed trajectory is unchanged.')
    parser.add_argument('--keep_going', action='store_true',
                        help='keep replaying after a failed action instead of stopping')
    args = parser.parse_args()

    with open(args.traj_json) as f:
        traj_data = json.load(f)

    render(traj_data, args.out_dir, args)
