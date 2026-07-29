#!/usr/bin/env python3
"""Retrofit LookUp/LookDown actions into an already-generated trajectory.

`generate_traj.py --look_around` bakes these in at generation time, but existing
trajectories were produced without them and there is no reason to regenerate a
2000-step episode just to change where the agent is looking while it walks.

The inserted pairs are balanced and never straddle an interaction, so the agent
arrives at every Pickup/Put/Open with exactly the horizon the planner chose --
the trajectory remains valid. Validate the result by replaying it:

    python env_utils/render_traj.py --traj_json <out.json> --out_dir /tmp/check ...

which reports "### replay diverged" if anything no longer reproduces.

Usage:
    python scripts/add_look_around.py in.json out.json [--min_run 6]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env_utils"))

NAV_ONLY = {"MoveAhead", "RotateLeft", "RotateRight"}
HORIZON_ADJ = 30


def insert_look_around(low_actions, min_run=6, base_horizon=30):
    """Standalone copy of gen_util.insert_look_around (importing gen_util pulls in
    ai2thor, which is python3.6-only; this script runs anywhere)."""
    out = []
    horizon = base_horizon
    holding = False
    i, n = 0, len(low_actions)
    while i < n:
        act = low_actions[i]
        name = act["api_action"]["action"]
        out.append(act)
        if name in ("LookUp", "LookDown"):
            horizon += -HORIZON_ADJ if name == "LookUp" else HORIZON_ADJ
        elif name == "PickupObject":
            holding = True
        elif name == "PutObject":
            holding = False

        j = i + 1
        while j < n and low_actions[j]["api_action"]["action"] in NAV_ONLY:
            j += 1
        run = j - i - 1

        # Look up to a level view (horizon 0) and back. AI2THOR clamps the camera
        # to [-30, 60]; restoring with a matching count keeps both ends legal even
        # if this static model drifts from the simulator by one step.
        # Never look around while carrying something: AI2THOR collision-checks the
        # held object and its height follows the camera horizon, so raising the head
        # mid-carry pushes it into walls and breaks navigation that previously worked.
        n_up = int(horizon // HORIZON_ADJ)
        if run >= min_run and 1 <= n_up <= 2 and j < n and not holding:
            hi = act.get("high_idx", 0)
            for _ in range(n_up):
                out.append({"api_action": {"action": "LookUp", "renderImage": True},
                            "high_idx": hi, "inserted": "look_around"})
            out.extend(low_actions[i + 1:j])
            for _ in range(n_up):
                out.append({"api_action": {"action": "LookDown", "renderImage": True},
                            "high_idx": low_actions[j].get("high_idx", hi),
                            "inserted": "look_around"})
            i = j
            continue
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min_run", type=int, default=6,
                    help="minimum navigation actions before it is worth lifting the head")
    args = ap.parse_args()

    traj = json.loads(Path(args.src).read_text())
    low = traj["plan"]["low_actions"]
    base = traj["scene"]["init_action"].get("horizon", 30)

    new_low = insert_look_around(low, min_run=args.min_run, base_horizon=base)

    # sub_trajs index into low_actions by range, so shift every boundary by the
    # number of insertions that happened before it
    shift_at = []
    added = 0
    orig_i = 0
    for a in new_low:
        if a.get("inserted") == "look_around":
            added += 1
        else:
            shift_at.append(added)
            orig_i += 1
    shift_at.append(added)

    for st in traj.get("sub_trajs", []):
        lo, hi = st["low_pddl_idx"]
        st["low_pddl_idx"] = [lo + shift_at[lo], hi + shift_at[min(hi, len(shift_at) - 1)]]

    for idx, a in enumerate(new_low):
        a["low_idx"] = idx
    traj["plan"]["low_actions"] = new_low
    traj.setdefault("gen_info", {})["look_around_retrofit"] = {
        "added": added, "min_run": args.min_run}

    Path(args.dst).write_text(json.dumps(traj))
    print("%d -> %d actions (+%d Look), %d subgoal ranges shifted"
          % (len(low), len(new_low), added, len(traj.get("sub_trajs", []))))


if __name__ == "__main__":
    main()
