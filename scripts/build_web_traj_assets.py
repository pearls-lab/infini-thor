#!/usr/bin/env python3
"""Turn rendered infini-THOR trajectories into the assets the project website plays.

Input  : one or more directories produced by `env_utils/render_traj.py`
         (each holds frames/%06d.jpg + timeline.json)
Output : docs/assets/video/<id>.mp4          h264, web-optimised, loops seamlessly
         docs/assets/img/traj/<id>.jpg       poster frame
         docs/_data/trajectories.json        metadata Jekyll inlines into the page

Usage
-----
    python scripts/build_web_traj_assets.py \
        --hero  /data/bkim/infini-thor-gen/render/fp230_a \
        --gallery /data/bkim/infini-thor-gen/render/fp210 \
                  /data/bkim/infini-thor-gen/render/fp323 \
        --fps 20
"""

import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def encode(render_dir: Path, out_mp4: Path, fps: int, height: int, crf: int):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # -vf scale keeps the aspect ratio and forces even dimensions (h264 requirement).
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(render_dir / "frames" / "%06d.jpg"),
        "-vf", "scale=-2:%d:flags=lanczos" % height,
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.0",
        "-movflags", "+faststart",
        "-an",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return out_mp4.stat().st_size


def poster(render_dir: Path, out_jpg: Path, timeline: dict):
    """Pick a frame a little way into the trajectory -- frame 0 often faces a wall."""
    idx = min(timeline["n_frames"] - 1, max(1, timeline["n_frames"] // 7))
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    src = render_dir / "frames" / ("%06d.jpg" % idx)
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-vf", "scale=-2:540", "-q:v", "4", str(out_jpg)],
        check=True,
    )


def filmstrip(render_dir: Path, out_png: Path, tiles: int = 14,
              tile_w: int = 152, height: int = 264):
    """Filmstrip of evenly spaced keyframes spanning the whole episode.

    (A true 1-px-per-step slit scan just smears each frame into a vertical
    stripe -- recognizable keyframes communicate the episode far better while
    keeping the same linear x -> t mapping the scrubber relies on.)"""
    frames = sorted((render_dir / "frames").glob("*.jpg"))
    n = len(frames)
    if n == 0:
        return None
    # Prefer interaction moments inside each tile's step range: a Pickup/Put frame
    # shows the manipulated object, where a mid-navigation frame often shows floor.
    interesting = set()
    tl_path = render_dir / "timeline.json"
    if tl_path.exists():
        tl = json.loads(tl_path.read_text())
        # step index -> frame index (identity+1 when rendered without smoothing)
        sf = tl.get("step_frames") or [i + 1 for i in range(tl["n_steps"])]
        interesting = {sf[st["t"]] for st in tl.get("steps", [])
                       if st["t"] < len(sf) and
                       st["a"] not in ("MoveAhead", "RotateRight", "RotateLeft",
                                       "LookUp", "LookDown", "Teleport", "TeleportFull")}

    strip = Image.new("RGB", (tiles * tile_w, height), (6, 7, 10))
    crop_aspect = tile_w / height
    for i in range(tiles):
        lo = round(i / tiles * (n - 1))
        hi = round((i + 1) / tiles * (n - 1))
        mid = (lo + hi) // 2
        # closest interaction frame to the tile centre, else the centre itself
        cands = [t for t in interesting if lo <= t <= hi]
        idx = min(cands, key=lambda t: abs(t - mid)) if cands else mid
        f = Image.open(frames[idx]).convert("RGB")
        cw = int(f.height * crop_aspect)
        x0 = (f.width - cw) // 2
        tile = f.crop((x0, 0, x0 + cw, f.height)).resize((tile_w, height), Image.LANCZOS)
        strip.paste(tile, (i * tile_w, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_png, optimize=True)
    return out_png


FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"]


def _font(name, size):
    for d in FONT_DIRS:
        f = Path(d) / name
        if f.exists():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


def social_card(render_dir: Path, timeline: dict, out_jpg: Path):
    """1200x630 Open Graph card: a frame from the run, dimmed, with the headline numbers."""
    frames = sorted((render_dir / "frames").glob("*.jpg"))
    if not frames:
        return
    src = Image.open(frames[len(frames) // 5]).convert("RGB")
    W, H = 1200, 630
    scale = max(W / src.width, H / src.height)
    src = src.resize((int(src.width * scale) + 1, int(src.height * scale) + 1), Image.LANCZOS)
    card = src.crop((0, 0, W, H))

    scrim = Image.new("L", (W, H))
    d = ImageDraw.Draw(scrim)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=int(80 + 140 * (y / H) ** 1.4))
    card = Image.composite(Image.new("RGB", (W, H), (6, 7, 10)), card, scrim)

    d = ImageDraw.Draw(card)
    d.text((72, 232), "\u221e-THOR", font=_font("DejaVuSans-Bold.ttf", 104), fill=(233, 236, 243))
    d.text((78, 362), "Beyond Needle(s) in the Embodied Haystack",
           font=_font("DejaVuSans.ttf", 38), fill=(200, 208, 222))
    tag = "%s  \u00b7  %d steps  \u00b7  %d subgoals  \u00b7  one continuous episode" % (
        timeline["scene"], timeline["n_steps"], timeline["n_subgoals"])
    d.text((78, 438), tag, font=_font("DejaVuSansMono.ttf", 25), fill=(124, 245, 213))
    d.text((78, 520), "arXiv 2505.16928   \u00b7   PEARLS Lab, UC San Diego",
           font=_font("DejaVuSansMono.ttf", 22), fill=(152, 162, 179))
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    card.save(out_jpg, quality=90)


def descriptor(render_dir: Path, timeline: dict, video_rel: str, poster_rel: str,
               tokens_per_step: int):
    """Compact per-trajectory metadata. Subgoal boundaries drive the HUD ticker, so we
    keep those; the full per-step list stays out of the page (it is large and unused)."""
    subgoals = [{"t": sg["start"], "text": sg["text"]} for sg in timeline["subgoals"]]
    n_steps = timeline["n_steps"]
    return {
        "id": render_dir.name,
        "scene": timeline["scene"],
        "steps": n_steps,
        "subgoals": subgoals,
        "n_subgoals": len(subgoals),
        "objects": timeline["n_objects_seen"],
        "tokens": n_steps * tokens_per_step,
        "step_frames": timeline.get("step_frames") or [],
        "n_frames": timeline.get("n_frames", n_steps + 1),
        "video": video_rel,
        "poster": poster_rel,
        "barcode": "assets/img/traj/%s_strip.png" % render_dir.name,
        "task": (timeline.get("long_horizon_task") or {}).get("task_desc", ""),
    }


def build(render_dir: Path, args, height: int, crf: int):
    timeline = json.loads((render_dir / "timeline.json").read_text())
    tid = render_dir.name
    video_rel = "assets/video/%s.mp4" % tid
    poster_rel = "assets/img/traj/%s.jpg" % tid

    size = encode(render_dir, REPO / "docs" / video_rel, args.fps, height, crf)
    poster(render_dir, REPO / "docs" / poster_rel, timeline)
    filmstrip(render_dir, REPO / "docs" / "assets" / "img" / "traj" / (tid + "_strip.png"))
    print("  %-28s %5d steps  %4d subgoals  %6.1f MB  %s"
          % (tid, timeline["n_steps"], timeline["n_subgoals"], size / 1e6, video_rel))
    return descriptor(render_dir, timeline, video_rel, poster_rel, args.tokens_per_step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hero", type=Path, required=True,
                   help="render dir for the trajectory that plays behind the intro")
    p.add_argument("--gallery", type=Path, nargs="*", default=[],
                   help="render dirs shown in the trajectory gallery")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--hero_height", type=int, default=720)
    p.add_argument("--gallery_height", type=int, default=480)
    p.add_argument("--crf", type=int, default=26)
    p.add_argument("--tokens_per_step", type=int, default=1450,
                   help="approx. LLM tokens per interleaved state+action step. The default "
                        "matches the context-size axis reported in the paper (~1.45K/step).")
    args = p.parse_args()

    print("building web assets ->", REPO / "docs")
    hero = build(args.hero, args, args.hero_height, args.crf)
    social_card(args.hero, json.loads((args.hero / "timeline.json").read_text()),
                REPO / "docs" / "assets" / "img" / "og-card.jpg")
    gallery = [build(d, args, args.gallery_height, args.crf + 2) for d in args.gallery]

    data = {
        "hero": hero,
        "gallery": gallery,
        "totals": {
            "max_steps": max([hero["steps"]] + [g["steps"] for g in gallery]),
            "max_tokens": max([hero["tokens"]] + [g["tokens"] for g in gallery]),
        },
    }
    out = REPO / "docs" / "_data" / "trajectories.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
