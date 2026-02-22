#!/usr/bin/env python3
"""
Needle(s) in the Embodied Haystack (NiEH) — Static Evaluation Script.

Supports multiple evaluation modes:
  - full_traj:      Feed the entire trajectory image sequence to the model.
  - haystack:       Build a controlled haystack context at varying depths (default when --full_traj is not set).
  - clip_retrieval:  Retrieve top-K images via CLIP similarity before prompting.
  - truncate_head:  Keep only the tail of the trajectory that fits in --ctx_size.
  - interleaved:    Interleave state images with action text from trajectory data.
  - text_state:     Use a text state summary + last frame.
  - video:          Pass the trajectory as a video file / frame list.
"""

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

import utils.nieh_utils as nieh_utils

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Estimated per-image token counts (300x300 images).
# Override with --n_img_token if your resolution or model differs.
MODEL_IMG_TOKEN_COUNTS = {
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": 1485,
    "Qwen/Qwen2.5-VL-7B-Instruct": 121,
    "deepseek-ai/deepseek-vl-7b-chat": 576,
}

MODEL_CLS_REGISTRY = {
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": LlavaOnevisionForConditionalGeneration,
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration,
}

DEPTH_MAP = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_cls(model_name_or_path: str):
    if model_name_or_path in MODEL_CLS_REGISTRY:
        return MODEL_CLS_REGISTRY[model_name_or_path]
    if "llava" in model_name_or_path.lower():
        return LlavaOnevisionForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


def apply_rope_scaling(
    cfg: AutoConfig,
    ctx_extension: Optional[str],
    factor: Optional[float],
) -> AutoConfig:
    """Apply RoPE scaling configuration for context extension."""
    if not ctx_extension:
        return cfg

    text_cfg = getattr(cfg, "text_config", cfg)
    mpe = getattr(text_cfg, "max_position_embeddings", None)

    if ctx_extension == "longrope":
        rope = {
            "rope_type": ctx_extension,
            "long_factor": factor,
            "short_factor": 1,
            "factor": 1.0,
            "original_max_position_embeddings": mpe,
        }
    else:
        rope = {
            "rope_type": ctx_extension,
            "factor": factor,
            "original_max_position_embeddings": mpe,
        }

    try:
        text_cfg.rope_scaling = rope
    except Exception:
        cfg.rope_scaling = rope

    return cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name_or_path: str,
    device: torch.device,
    dtype=torch.bfloat16,
    ctx_extension: Optional[str] = None,
    ctx_extension_factor: Optional[float] = None,
    attn_impl: str = "flash_attention_2",
    base_model: Optional[str] = None,
):
    """Load a VLM from a HuggingFace hub model or a local DCP checkpoint."""
    is_local = os.path.exists(model_name_or_path)
    model_cls = get_model_cls(base_model or model_name_or_path)

    if is_local:
        # Local distributed-checkpoint path — requires --base_model
        cfg = AutoConfig.from_pretrained(
            base_model,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        model = model_cls(cfg)
        model.to(device=device, dtype=dtype)
        state = {"model": model.state_dict()}
        dcp.load(state, checkpoint_id=model_name_or_path)
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Standard HuggingFace hub loading
        logger.info(f"Loading from HuggingFace hub: {model_name_or_path}")
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if ctx_extension:
            logger.info(f"Applying context extension: {ctx_extension} (factor={ctx_extension_factor})")
            cfg = apply_rope_scaling(cfg, ctx_extension, ctx_extension_factor)

        model = model_cls.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            config=cfg,
            attn_implementation=attn_impl,
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    model.eval()
    return model, cfg, processor


# ---------------------------------------------------------------------------
# CLIP retrieval helpers
# ---------------------------------------------------------------------------

def load_clip_retriever(model_name: str, device: torch.device):
    """Load a CLIP model and processor for image retrieval."""
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.to(device).eval()
    return clip_model, clip_processor


def retrieve_top_images_clip(
    question: str,
    img_list: List[Image.Image],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    top_k: int = 10,
) -> Tuple[List[Image.Image], List[int]]:
    """Retrieve top-k images most similar to the question using CLIP.
    Returns selected images in chronological order and their indices."""
    if not img_list:
        return [], []

    text_inputs = clip_processor(text=[question], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    all_img_feats = []
    batch_size = 32
    for start in range(0, len(img_list), batch_size):
        batch_imgs = img_list[start : start + batch_size]
        img_inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = clip_model.get_image_features(**img_inputs)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        all_img_feats.append(img_feats)

    img_features = torch.cat(all_img_feats, dim=0)
    sims = (img_features @ text_features.T).squeeze(1)

    k = min(top_k, len(img_list))
    _, top_indices = torch.topk(sims, k=k, largest=True, sorted=True)
    chronological_indices = sorted(top_indices.tolist())
    selected_imgs = [img_list[i] for i in chronological_indices]
    return selected_imgs, chronological_indices


def truncate_head_by_ctx_size(
    img_list: List[Image.Image],
    ctx_size_k: int,
    n_img_token: int,
) -> Tuple[List[Image.Image], List[int]]:
    """Keep only the tail of the trajectory that fits within ctx_size_k * 1024 tokens."""
    if not img_list:
        return [], []
    if not n_img_token or n_img_token <= 0:
        return img_list, list(range(len(img_list)))

    max_images = max(1, (ctx_size_k * 1024) // n_img_token)
    start_idx = max(0, len(img_list) - max_images)
    indices = list(range(start_idx, len(img_list)))
    return img_list[start_idx:], indices


def remove_plan_tags(text: str) -> str:
    """Remove all <|plan|>...<|plan|> spans from text."""
    return re.sub(r"<\|plan\|>.*?<\|plan\|>", "", text)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    img_list: List[Image.Image],
    video_path: Optional[str],
    prompt_messages: List[Dict],
    processor: Any,
    model: Any,
    device: torch.device,
) -> Tuple[str, Optional[int]]:
    """Run VLM generation and return (response_text, input_token_count)."""
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device, torch.bfloat16)

    ctx_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None
    out = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.eos_token_id)

    decoded = processor.batch_decode(out, skip_special_tokens=True)
    lm_response = decoded[0].strip().split("\n")[-1]
    return lm_response, ctx_len


# ---------------------------------------------------------------------------
# Eval-mode context builders
# ---------------------------------------------------------------------------

def _build_task_instruction(eval_mode: str, question: str) -> str:
    """Return the task instruction string for a given eval mode."""
    prefix_map = {
        "interleaved": (
            "\nAnswer the question given the state and action sequence of the embodied agent. "
            "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
        ),
        "text_state": (
            "\nAnswer the question given the description of the environment and the agent's state (the last image). "
            "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
        ),
        "video": (
            "\nAnswer the question given the video showing the agent's trajectory over time. "
            "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
        ),
    }
    default = (
        "\nAnswer the question given the agent's views in time order. "
        "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
    )
    return prefix_map.get(eval_mode, default) + question


def _build_full_traj_context(
    eval_mode: str,
    row: Dict,
    img_list: List[Image.Image],
    lowidx2imgs: Dict,
    task_instr: str,
    metadata_dir: str,
    traj_dir: Optional[str],
    n_img_token: int,
    ctx_size: int,
    clip_model: Optional[CLIPModel],
    clip_processor: Optional[CLIPProcessor],
    clip_top_k: int,
    device: torch.device,
) -> Tuple[List[Image.Image], List[Dict], Optional[str], List[int]]:
    """Build (ctx_img_list, messages, video_path, selected_indices) for --full_traj modes."""
    selected_indices = list(range(len(img_list)))
    video_path = None
    ctx_img_list = []

    if eval_mode == "video":
        video_path_str = os.path.join(metadata_dir, row["traj_id"], f"{row['traj_id']}_10fps.mp4")
        if not os.path.exists(video_path_str):
            logger.warning(f"Video not found: {video_path_str}, falling back to frames.")
            content = [{"type": "video", "video": img_list}]
        else:
            content = [
                {"type": "video", "video": video_path_str, "fps": 1.0},
                {"type": "text", "text": task_instr},
            ]
        messages = [{"role": "user", "content": content}]

    elif eval_mode == "clip_retrieval":
        if clip_model is None or clip_processor is None:
            raise RuntimeError("eval_mode='clip_retrieval' requires --clip_model_name or a default CLIP model.")
        ctx_img_list, selected_indices = retrieve_top_images_clip(
            question=row["question"],
            img_list=img_list,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            top_k=clip_top_k,
        )
        content = [{"type": "image", "image": im} for im in ctx_img_list] + [{"type": "text", "text": task_instr}]
        messages = [{"role": "user", "content": content}]

    elif eval_mode == "truncate_head":
        ctx_img_list, selected_indices = truncate_head_by_ctx_size(img_list, ctx_size, n_img_token)
        content = [{"type": "image", "image": im} for im in ctx_img_list] + [{"type": "text", "text": task_instr}]
        messages = [{"role": "user", "content": content}]

    elif eval_mode == "interleaved":
        traj_data = _load_traj_json(traj_dir, row["traj_id"])
        content = []
        for sub_task, sub_traj in zip(traj_data["sub_tasks"], traj_data["sub_trajs"]):
            content.append({"type": "text", "text": f"Your task goal: {sub_task['task_desc']}. "})
            low_start, low_end = sub_traj["low_pddl_idx"]

            # Initial state image
            content.append({"type": "text", "text": " State: "})
            if low_start == 0:
                init_img = img_list[0]
            else:
                init_img = lowidx2imgs[low_start - 1][-1]
            content.append({"type": "image", "image": init_img})
            ctx_img_list.append(init_img)

            for low_idx in range(low_start, low_end):
                act_dict = traj_data["plan"]["low_actions"][low_idx]["api_action"]
                act_str = nieh_utils.act_dict_to_str(act_dict)
                content.append({"type": "text", "text": f" Action: {act_str}"})
                if lowidx2imgs.get(low_idx):
                    content.append({"type": "text", "text": " State: "})
                    for low_img in lowidx2imgs[low_idx]:
                        content.append({"type": "image", "image": low_img})
                        ctx_img_list.append(low_img)

        content.append({"type": "text", "text": task_instr})
        messages = [{"role": "user", "content": content}]

    elif eval_mode == "text_state":
        traj_data = _load_traj_json(traj_dir, row["traj_id"])
        last_low_idx = len(traj_data["plan"]["low_actions"]) - 1
        state_summary = traj_data["state_summary"][str(last_low_idx)]
        ctx_img_list = [img_list[-1]]
        content = [
            {"type": "text", "text": f" State: {state_summary}; Last agent's view: "},
            {"type": "image", "image": ctx_img_list[0]},
            {"type": "text", "text": task_instr},
        ]
        messages = [{"role": "user", "content": content}]

    else:
        # Default: full_traj — feed all images
        ctx_img_list = img_list
        content = [{"type": "image", "image": im} for im in ctx_img_list] + [{"type": "text", "text": task_instr}]
        messages = [{"role": "user", "content": content}]

    return ctx_img_list, messages, video_path, selected_indices


def _load_traj_json(traj_dir: str, traj_id: str) -> Dict:
    """Load trajectory JSON from traj_dir, trying .json then .txt extensions."""
    for ext in (".json", ".txt"):
        path = os.path.join(traj_dir, f"{traj_id}{ext}")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"Trajectory file not found for {traj_id} in {traj_dir}")


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

def _build_log_path(
    qa_file_path: Optional[str],
    model_name_or_path: str,
    full_traj: bool,
    eval_mode: str,
    ctx_size: int,
    clip_top_k: int,
) -> str:
    """Construct a descriptive log file path under output/."""
    qa_base = os.path.splitext(os.path.basename(qa_file_path))[0] if qa_file_path else "qa"
    model_base = os.path.basename(model_name_or_path.rstrip("/")).replace(":", "_")
    parts = [f"eval_{qa_base}_{model_base}"]
    if full_traj:
        parts.append("full_traj")
    if full_traj and eval_mode and eval_mode != "full_traj":
        parts.append(eval_mode)
    if eval_mode == "truncate_head":
        parts.append(f"{ctx_size}K")
    if eval_mode == "clip_retrieval":
        parts.append(f"top{clip_top_k}")
    os.makedirs("output", exist_ok=True)
    return os.path.join("output", "_".join(parts) + ".log")


def _load_existing_results(log_path: str, full_traj: bool, target_depths: List[int]):
    """Resume from an existing JSONL log file. Returns (existing_qidx_set, stats)."""
    existing_qidx = set()
    total_match = 0.0 if full_traj else [0.0 for _ in target_depths]
    total_count = 0.0 if full_traj else [0.0 for _ in target_depths]
    oe_match, oe_count = 0.0, 0.0

    if not os.path.exists(log_path):
        return existing_qidx, total_match, total_count, oe_match, oe_count

    logger.info(f"Resuming from existing log: {log_path}")
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = rec.get("qidx")
            if qid is None:
                continue
            existing_qidx.add(int(qid))

            if full_traj:
                score_val = rec.get("score")
                if score_val is not None:
                    total_match += float(score_val)
                    total_count += 1.0
                    gt_answer = rec.get("gt_answer", [])
                    if gt_answer and isinstance(gt_answer[0], str) and gt_answer[0].lower() not in ("yes", "no"):
                        oe_match += float(score_val)
                        oe_count += 1.0
            else:
                for item in rec.get("per_depth", []):
                    try:
                        depth = int(item["depth"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if 0 <= depth < len(total_match):
                        total_match[depth] += float(item.get("score", 0))
                        total_count[depth] += float(item.get("count", 1.0))

    return existing_qidx, total_match, total_count, oe_match, oe_count


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def main(
    qa_data: List[Dict[str, Any]],
    metadata_dir: str,
    model_name_or_path: str,
    ctx_size: int = 32,
    target_depths: List[int] = [0, 1, 2, 3, 4],
    ctx_extension: Optional[str] = None,
    ctx_extension_factor: Optional[float] = None,
    full_traj: bool = False,
    attn_impl: str = "flash_attention_2",
    base_model: Optional[str] = None,
    n_img_token_override: Optional[int] = None,
    qa_file_path: Optional[str] = None,
    traj_dir: Optional[str] = None,
    eval_mode: str = "full_traj",
    clip_model_name: Optional[str] = None,
    clip_top_k: int = 10,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load model ---
    logger.info("Loading model...")
    model, _, processor = load_model(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=torch.bfloat16,
        ctx_extension=ctx_extension,
        ctx_extension_factor=ctx_extension_factor,
        attn_impl=attn_impl,
        base_model=base_model,
    )

    tokenizer = processor.tokenizer
    tokenizer.model_max_length = max(getattr(tokenizer, "model_max_length", 0) or 0, 1_048_576)

    # --- Resolve per-image token count ---
    if n_img_token_override is not None:
        n_img_token = n_img_token_override
    else:
        model_key = base_model or model_name_or_path
        n_img_token = MODEL_IMG_TOKEN_COUNTS.get(model_key)
    if n_img_token is None:
        raise ValueError(
            f"Unknown image token count for '{base_model or model_name_or_path}'. "
            f"Use --n_img_token to specify the per-image token count."
        )

    # --- Optional CLIP retriever ---
    clip_model, clip_proc = None, None
    if full_traj and eval_mode == "clip_retrieval":
        clip_model_name = clip_model_name or "openai/clip-vit-large-patch14"
        logger.info(f"Loading CLIP model: {clip_model_name}")
        clip_model, clip_proc = load_clip_retriever(clip_model_name, device)

    # --- Resume / logging setup ---
    log_path = _build_log_path(qa_file_path, model_name_or_path, full_traj, eval_mode, ctx_size, clip_top_k)
    existing_qidx, total_match, total_count, oe_match, oe_count = _load_existing_results(
        log_path, full_traj, target_depths
    )
    logger.info(f"Writing results to: {log_path}")
    log_f = open(log_path, "a", encoding="utf-8")

    # --- Evaluation loop ---
    for qidx, row in enumerate(qa_data):
        if qidx in existing_qidx:
            logger.info(f"Skipping qidx={qidx} (already completed)")
            continue

        img_list, metadata, traj_text, img_path_list, lowidx2imgs = nieh_utils.load_qa_data(
            row["traj_id"], metadata_dir
        )
        assert len(img_list) == len(metadata["img_idx"]), (
            f"Image count mismatch for trajectory {row['traj_id']}"
        )

        task_instr = _build_task_instruction(eval_mode, row["question"])

        if full_traj:
            ctx_img_list, messages, video_path, selected_indices = _build_full_traj_context(
                eval_mode=eval_mode,
                row=row,
                img_list=img_list,
                lowidx2imgs=lowidx2imgs,
                task_instr=task_instr,
                metadata_dir=metadata_dir,
                traj_dir=traj_dir,
                n_img_token=n_img_token,
                ctx_size=ctx_size,
                clip_model=clip_model,
                clip_processor=clip_proc,
                clip_top_k=clip_top_k,
                device=device,
            )

            logger.info(
                f"[qidx={qidx}] gt_idx: {row['gt_img_idx']}, mode: {eval_mode}, n_imgs: {len(ctx_img_list)}"
            )

            lm_response, ctx_len = generate(ctx_img_list, video_path, messages, processor, model, device)
            if lm_response:
                score = nieh_utils.get_score(lm_response, row["answer"])
                logger.info(
                    f"  score: {score}, response: {lm_response}, answer: {row['answer']}"
                )
                total_match += score
                total_count += 1.0

                if isinstance(row["answer"][0], str) and row["answer"][0].lower() not in ("yes", "no"):
                    oe_match += score
                    oe_count += 1.0

                record = {
                    "qidx": qidx,
                    "traj_id": row.get("traj_id"),
                    "question": row.get("question"),
                    "gt_img_idx": row.get("gt_img_idx"),
                    "ctx_size_k": ctx_size,
                    "n_imgs": len(ctx_img_list),
                    "ctx_n_tokens": ctx_len,
                    "llm_response": lm_response,
                    "gt_answer": row.get("answer"),
                    "score": float(score),
                    "eval_mode": eval_mode,
                    "selected_img_indices": selected_indices,
                }
                json.dump(record, log_f, ensure_ascii=False)
                log_f.write("\n")
                log_f.flush()
        else:
            # Haystack mode: evaluate at multiple needle depths
            NiH_match = [0.0 for _ in target_depths]
            NiH_count = [0.0 for _ in target_depths]
            per_depth_results = []

            for di, depth in enumerate(target_depths):
                ctx_img_list, _ = nieh_utils.build_haystack(
                    ctx_size, depth, row["gt_img_idx"], n_img_token, img_list
                )
                if not ctx_img_list:
                    continue

                # Build messages for haystack context
                content = [{"type": "image", "image": im} for im in ctx_img_list]
                content.append({"type": "text", "text": task_instr})
                messages = [{"role": "user", "content": content}]

                lm_response, ctx_len = generate(ctx_img_list, None, messages, processor, model, device)
                if lm_response:
                    score = nieh_utils.get_score(lm_response, row["answer"])
                    logger.info(
                        f"[qidx={qidx}] ctx_size: {ctx_size}K, depth: {DEPTH_MAP[depth]}, "
                        f"n_imgs: {len(ctx_img_list)}, score: {score}, "
                        f"response: {lm_response}, answer: {row['answer']}"
                    )
                    NiH_match[di] = score
                    NiH_count[di] = 1.0
                    per_depth_results.append({
                        "depth": depth,
                        "depth_label": DEPTH_MAP[depth],
                        "n_imgs": len(ctx_img_list),
                        "ctx_n_tokens": ctx_len,
                        "llm_response": lm_response,
                        "gt_answer": row.get("answer"),
                        "score": float(score),
                        "count": 1.0,
                    })

            if per_depth_results:
                record = {
                    "qidx": qidx,
                    "traj_id": row.get("traj_id"),
                    "question": row.get("question"),
                    "gt_img_idx": row.get("gt_img_idx"),
                    "mode": "haystack",
                    "ctx_size_k": ctx_size,
                    "model_name": model_name_or_path,
                    "base_model": base_model,
                    "per_depth": per_depth_results,
                }
                json.dump(record, log_f, ensure_ascii=False)
                log_f.write("\n")
                log_f.flush()

            total_match = [x + y for x, y in zip(total_match, NiH_match)]
            total_count = [x + y for x, y in zip(total_count, NiH_count)]

    # --- Final results ---
    logger.info("\n===== Final Results =====")
    if full_traj:
        if total_count > 0:
            score = total_match / total_count
            logger.info(f"Overall score: {score:.4f}  (match={total_match}, count={total_count})")
        if oe_count > 0:
            oe_score = oe_match / oe_count
            logger.info(f"Open-ended score: {oe_score:.4f}  (match={oe_match}, count={oe_count})")
    else:
        for depth in target_depths:
            if total_count[depth] > 0:
                score = total_match[depth] / total_count[depth]
                logger.info(
                    f"ctx_size: {ctx_size}K, depth: {DEPTH_MAP[depth]}, "
                    f"score: {score:.4f}  (match={total_match[depth]}, count={total_count[depth]})"
                )

    log_f.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Needle(s) in the Embodied Haystack (NiEH) evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--qa_file_path", type=str, required=True, help="Path to the QA CSV file")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Directory containing metadata and images")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model name or local checkpoint path",
    )
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (required for local checkpoints)")
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation",
    )

    # Evaluation mode
    parser.add_argument("--full_traj", action="store_true", help="Use full trajectory instead of haystack building")
    parser.add_argument(
        "--eval_mode", type=str, default="full_traj",
        choices=["full_traj", "interleaved", "clip_retrieval", "truncate_head", "text_state", "video"],
        help="Evaluation mode (only used with --full_traj)",
    )
    parser.add_argument(
        "--traj_dir", type=str, default=None,
        help="Directory with trajectory JSON files (required for interleaved/text_state modes)",
    )

    # Context / haystack arguments
    parser.add_argument("--ctx_size", type=int, default=32, help="Context size in K tokens")
    parser.add_argument("--n_img_token", type=int, default=None, help="Override per-image token count")

    # Context extension (RoPE scaling)
    parser.add_argument("--ctx_extension", type=str, default=None, help="Context extension type (e.g., 'yarn', 'longrope')")
    parser.add_argument("--ctx_extension_factor", type=float, default=4.0, help="Context extension factor")

    # CLIP retrieval options
    parser.add_argument("--clip_model_name", type=str, default=None, help="CLIP model for retrieval mode")
    parser.add_argument("--clip_top_k", type=int, default=10, help="Number of top images to retrieve with CLIP")

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.qa_file_path):
        raise FileNotFoundError(f"QA file not found: {args.qa_file_path}")
    if args.eval_mode in ("interleaved", "text_state") and not args.traj_dir:
        parser.error(f"--traj_dir is required for eval_mode='{args.eval_mode}'")

    # Load QA data
    qa_data = []
    with open(args.qa_file_path, "r") as f:
        for row in csv.DictReader(f):
            row_dict = dict(row)
            if "gt_img_idx" in row_dict:
                row_dict["gt_img_idx"] = ast.literal_eval(row_dict["gt_img_idx"])
            if "answer" in row_dict:
                row_dict["answer"] = ast.literal_eval(row_dict["answer"])
            qa_data.append(row_dict)

    main(
        qa_data=qa_data,
        metadata_dir=args.metadata_dir,
        model_name_or_path=args.model_name_or_path,
        ctx_size=args.ctx_size,
        ctx_extension=args.ctx_extension,
        ctx_extension_factor=args.ctx_extension_factor,
        full_traj=args.full_traj,
        attn_impl=args.attn_impl,
        base_model=args.base_model,
        n_img_token_override=args.n_img_token,
        qa_file_path=args.qa_file_path,
        traj_dir=args.traj_dir,
        eval_mode=args.eval_mode,
        clip_model_name=args.clip_model_name,
        clip_top_k=args.clip_top_k,
    )
