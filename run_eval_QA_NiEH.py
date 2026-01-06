import argparse
import csv
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import logging
from PIL import Image

import torch
import torch.distributed.checkpoint as dcp
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)
from transformers.utils import is_flash_attn_2_available, is_torch_cuda_available
from qwen_vl_utils import process_vision_info
import utils.nieh_utils as nieh_utils

#os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# estimated/known image-token counts per 300x300 image.
# use --n_img_token to override if your resolution differs or the model isn’t listed.
model2num_img_token = {
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': 1485, # base_image_feature (27, 27) + image_feature (+ new line) (27, 28)
    'Qwen/Qwen2.5-VL-7B-Instruct': 121, # 121 tokens (for 300x300 images)
    'deepseek-ai/deepseek-vl-7b-chat': 576, # 576 tokens (for 300x300 images)
}

model_zoo = {
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': LlavaOnevisionForConditionalGeneration,
    'Qwen/Qwen2.5-VL-7B-Instruct': Qwen2_5_VLForConditionalGeneration
}

depth_map = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}



def remove_plan_tags(text):
    """
    Remove all text with patterns <|plan|>...<|plan|> from the input text.
    
    Args:
        text (str): Input text containing plan tags
        
    Returns:
        str: Text with all <|plan|>...<|plan|> patterns removed
    """
    # Use regex to find and remove <|plan|>...<|plan|> patterns
    # The pattern matches <|plan|>, any characters (non-greedy), and closing <|plan|>
    pattern = r'<\|plan\|>.*?<\|plan\|>'
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


def load_clip_retriever(model_name: str, device: torch.device):
    """
    Load a CLIP (or CLIP-like) model and processor for retrieval.
    """
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.to(device)
    clip_model.eval()
    return clip_model, clip_processor


def retrieve_top_images_clip(
    question: str,
    img_list: List[Image.Image],
    clip_model: "CLIPModel",
    clip_processor: "CLIPProcessor",
    device: torch.device,
    top_k: int = 10,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Retrieve top-k images most similar to the question using CLIP.
    Returns the selected images (kept in temporal order) and their indices.
    """
    if len(img_list) == 0:
        return [], []

    # Encode question
    text_inputs = clip_processor(
        text=[question],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Encode images in batches
    all_img_feats = []
    batch_size = 32
    for start in range(0, len(img_list), batch_size):
        batch_imgs = img_list[start:start + batch_size]
        img_inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = clip_model.get_image_features(**img_inputs)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        all_img_feats.append(img_feats)

    img_features = torch.cat(all_img_feats, dim=0)
    sims = (img_features @ text_features.T).squeeze(1)

    k = min(top_k, len(img_list))
    _, top_indices = torch.topk(sims, k=k, largest=True, sorted=True)
    top_indices_list = top_indices.tolist()

    # For the context, keep chronological order
    chronological_indices = sorted(top_indices_list)
    selected_imgs = [img_list[i] for i in chronological_indices]
    return selected_imgs, chronological_indices


def truncate_head_by_ctx_size(
    img_list: List[Image.Image],
    ctx_size_k: int,
    n_img_token: int,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Approximate truncation of the trajectory head, keeping only the tail that
    fits into the given context size (in K tokens) based on a per-image token estimate.
    """
    if len(img_list) == 0:
        return [], []

    if n_img_token is None or n_img_token <= 0:
        # Fallback: keep the full trajectory if we don't know the image token count
        indices = list(range(len(img_list)))
        return img_list, indices

    max_tokens = ctx_size_k * 1024  # ctx_size is expressed in K tokens
    max_images = max_tokens // n_img_token
    if max_images <= 0:
        max_images = 1

    if max_images >= len(img_list):
        start_idx = 0
    else:
        start_idx = len(img_list) - max_images

    indices = list(range(start_idx, len(img_list)))
    selected_imgs = img_list[start_idx:]
    return selected_imgs, indices


def get_model_cls(model_name_or_path):
    if model_name_or_path in model_zoo:
        return model_zoo[model_name_or_path]
    else:
        if 'llava' in model_name_or_path.lower():
            return LlavaOnevisionForConditionalGeneration
        else:
            return Qwen2_5_VLForConditionalGeneration


def apply_rope_scaling(cfg: AutoConfig, ctx_extension: Optional[str], factor: Optional[float]):
    if not ctx_extension:
        return cfg

    text_cfg = getattr(cfg, "text_config", cfg)
    
    if not hasattr(text_cfg, "max_position_embeddings"):    
        mpe = None
    else:
        mpe = text_cfg.max_position_embeddings

    # LongRoPE expects additional fields; others just need type + factor.
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


def load_model(model_name_or_path: str,
                  device: torch.device,
                  dtype=torch.bfloat16,
                  ctx_extension: Optional[str]=None,
                  ctx_extension_factor: Optional[float]=None,
                  attn_impl: str = "flash_attention_2",
                  base_model: str = None):
    
    is_local = os.path.exists(model_name_or_path)
    model_cls = get_model_cls(model_name_or_path)

    if is_local:
        cfg = AutoConfig.from_pretrained(
            base_model, 
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        model = model_cls(cfg)
        model.to(device=device, dtype=dtype)
        state = {"model": model.state_dict()}
        dcp.load(state, checkpoint_id=model_name_or_path)

        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Standard loading (HuggingFace format)
        print(f"Loading from {'local path' if is_local else 'HuggingFace hub'}: {model_name_or_path}")
        
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if ctx_extension:
            logger.info(f"Using dynamic context length: {ctx_extension} (factor={ctx_extension_factor})")
            cfg = apply_rope_scaling(cfg, ctx_extension, ctx_extension_factor)

        kwargs = dict(
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            config=cfg,
            attn_implementation=attn_impl,
            #attn_implementation="eager",
        )
        model = model_cls.from_pretrained(model_name_or_path, **kwargs)
        #model.to(device)

        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    model.eval()

    return model, cfg, processor


# @torch.no_grad()
# def generate(
#     img_list: List[Image.Image],
#     video_path: str,
#     prompt: str, 
#     processor: Any, 
#     model: Any, 
#     device: torch.device
# ) -> Tuple[str, Optional[int]]:
#     if video_path:
#         inputs = processor(videos=img_list, text=prompt, padding=True, return_tensors="pt").to(device, torch.bfloat16)
#     else:
#         inputs = processor(images=img_list, text=prompt, padding=True, return_tensors="pt").to(device, torch.bfloat16)
    
#     image_inputs, video_inputs = process_vision_info(prompt_messages)
    
#     ctx_len: Optional[int] = None
#     if "input_ids" in inputs:
#         try:
#             ctx_len = int(inputs["input_ids"].shape[1])
#         except Exception:
#             ctx_len = None
    
#     out = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.eos_token_id)
    
#     decoded_out = processor.batch_decode(out, skip_special_tokens=True)
#     lm_response = decoded_out[0].strip().split("\n")[-1]
#     return lm_response, ctx_len


@torch.no_grad()
def generate(
    img_list: List[Image.Image],
    video_path: Optional[str], # Now potentially a string path
    prompt_messages: List[Dict], # Pass the messages list instead of the formatted string
    processor: Any, 
    model: Any, 
    device: torch.device
) -> Tuple[str, Optional[int]]:
    
    # 1. Use the utility to process the messages (extracts frames from .mp4 or uses PIL list)
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    
    # 2. Apply chat template for the text part
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # 3. Prepare inputs
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device, torch.bfloat16)
    
    ctx_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None
    out = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.eos_token_id)
    
    decoded_out = processor.batch_decode(out, skip_special_tokens=True)
    lm_response = decoded_out[0].strip().split("\n")[-1]
    return lm_response, ctx_len


@torch.no_grad()
def main(
    qa_data: List[Dict[str, Any]],
    traj_dir: str,
    metadata_dir: str,
    model_name_or_path: str,
    ctx_size: int = 32,
    target_depths: List[int] = [0, 1, 2, 3, 4],
    ctx_extension: Optional[str] = None,
    ctx_extension_factor: Optional[float] = None,
    full_traj: bool = False,
    attn_impl: str = "flash_attention_2",
    base_model: str = None,
    qa_file_path: Optional[str] = None,
    eval_mode: str = "full_traj",
    clip_model_name: Optional[str] = None,
    clip_top_k: int = 10,
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Model
    logger.info("Loading model...")
    model, llm_config, processor = load_model(
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
    
    n_img_token = model2num_img_token[base_model] if base_model else model2num_img_token[model_name_or_path]
    if n_img_token is None:
        raise ValueError(f"Unknown image token count for model '{model_name_or_path}'. ")

    # Optional CLIP retriever (used only when full_traj is True and eval_mode == 'clip_retrieval')
    clip_model = None
    clip_processor = None
    if full_traj and eval_mode == "clip_retrieval":
        clip_model_name = clip_model_name or "openai/clip-vit-large-patch14"
        logger.info(f"Loading CLIP model for retrieval: {clip_model_name}")
        clip_model, clip_processor = load_clip_retriever(clip_model_name, device)

    if full_traj:
        total_match, total_count = 0.0, 0.0
        oe_match, oe_count = 0.0, 0.0
    else:
        total_match = [0 for _ in target_depths]
        total_count = [0 for _ in target_depths]

    # Prepare JSONL logging / resume from existing results
    qa_basename = os.path.splitext(os.path.basename(qa_file_path))[0] if qa_file_path is not None else "qa"
    model_basename = os.path.basename(model_name_or_path.rstrip("/"))
    os.makedirs("output", exist_ok=True)
    safe_qa = qa_basename.replace(os.sep, "_")
    safe_model = model_basename.replace(os.sep, "_").replace(":", "_")
    safe_full_traj = "_full_traj" if full_traj else ""
    safe_mode = f"_{eval_mode}" if (full_traj and eval_mode and eval_mode != "full_traj") else ""
    safe_ctx_size = f"_{ctx_size}K" if eval_mode == "truncate_head" else ""
    safe_topk = f"_top{clip_top_k}" if eval_mode == "clip_retrieval" else ""
    log_path = os.path.join(
        "output",
        f"eval_{safe_qa}_{safe_model}{safe_full_traj}{safe_mode}.log",
    )
    log_path = os.path.join("output", f"eval_{safe_qa}_{safe_model}{safe_full_traj}{safe_mode}{safe_ctx_size}{safe_topk}.log")

    existing_qidx = set()
    if os.path.exists(log_path):
        logger.info(f"Found existing log file at {log_path}, will resume and skip completed questions.")
        with open(log_path, "r", encoding="utf-8") as f_log:
            for line in f_log:
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
                try:
                    qid_int = int(qid)
                except (TypeError, ValueError):
                    continue
                existing_qidx.add(qid_int)

                # Warm start aggregate stats from existing records
                if full_traj:
                    score_val = rec.get("score")
                    if score_val is not None:
                        total_match += float(score_val)
                        total_count += 1.0

                        if isinstance(rec.get("gt_answer")[0], str) and not rec.get("gt_answer")[0].lower() in ["no", "yes"]:
                            oe_match += float(score_val)
                            oe_count += 1.0
                else:
                    per_depth = rec.get("per_depth", [])
                    for item in per_depth:
                        try:
                            depth = int(item.get("depth"))
                        except (TypeError, ValueError):
                            continue
                        if depth < 0 or depth >= len(total_match):
                            continue
                        score_val = item.get("score")
                        if score_val is None:
                            continue
                        count_val = float(item.get("count", 1.0))
                        total_match[depth] += float(score_val)
                        total_count[depth] += count_val

    logger.info(f"Writing per-question results to: {log_path}")
    log_f = open(log_path, "a", encoding="utf-8")

    for qidx, row in enumerate(qa_data):
        if qidx in existing_qidx:
            logger.info(f"Skipping qidx={qidx} (already present in {log_path})")
            continue

        img_list, metadata, traj_text, img_path_list, lowidx2imgs = nieh_utils.load_qa_data(row['traj_id'], metadata_dir)

        assert len(img_list) == len(metadata['img_idx']), f"Image count mismatch for trajectory {row['traj_id']}"

        if eval_mode == "interleaved":
            task_instr = (
                "\nAnswer the question given the state and action sequence of the embodied agent. "
                "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
            )
        elif eval_mode == "text_state":
            task_instr = (
                "\nAnswer the question given the description of the environment and the agent's state (the last image). "
                "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
            )
        elif eval_mode == "video":
            task_instr = (
                "\nAnswer the question given the video showing the agent's trajectory over time. "
                "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
            )
        else:
            task_instr = (
                "\nAnswer the question given the agent's views in time order. "
                "Do not include explanation or reasoning in the answer. Answer with a single word or words.\nQuestion: "
            )

        task_instr += row['question']
        # prompt = (
        #     "You are given a sequence of ego-centric images from an embodied agent. "
        #     "They are ordered in time from earliest to latest. "
        #     "Use only the information in these images to answer the question.\n\n"
        #     f"Question: {row['question']}\n\n"
        #     "Answer guidelines:\n"
        #     "- Respond with the final answer only, without explanation or reasoning.\n"
        #     "- If the answer is yes/no, reply with exactly 'yes' or 'no'.\n"
        #     "- If the answer is a number or count, reply using digits only (e.g., '2').\n"
        # )

        if full_traj:
            # Select context images based on eval_mode
            selected_indices = list(range(len(img_list)))
            video_path = None
            ctx_img_list = []
            if eval_mode == "video":
                # Construct the actual path
                video_path_str = os.path.join(metadata_dir, row['traj_id'], f"{row['traj_id']}_10fps.mp4")
                
                # Check if file exists
                if not os.path.exists(video_path_str):
                    logger.warning(f"Video file not found: {video_path_str}, falling back to frames.")
                    content = [{'type': 'video', 'video': img_list}]
                else:
                    # Use the .mp4 path directly
                    content = [
                        {
                            "type": "video",
                            "video": video_path_str,
                            "fps": 1.0, # Adjust sampling frequency as needed
                        },
                        {"type": "text", "text": task_instr}
                    ]
                messages = [{"role": "user", "content": content}]             
            elif eval_mode == "clip_retrieval":
                if clip_model is None or clip_processor is None:
                    raise RuntimeError("eval_mode='clip_retrieval' requires a loaded CLIP model.")
                ctx_img_list, selected_indices = retrieve_top_images_clip(
                    question=row["question"],
                    img_list=img_list,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=device,
                    top_k=clip_top_k,
                )
                content = [{'type': 'image'} for _ in ctx_img_list] + [{'type': 'text', 'text': task_instr}]
                messages = [{"role": "user", "content": content}]
            elif eval_mode == "truncate_head":
                ctx_img_list, selected_indices = truncate_head_by_ctx_size(
                    img_list=img_list,
                    ctx_size_k=ctx_size,
                    n_img_token=n_img_token,
                )
                content = [{'type': 'image'} for _ in ctx_img_list] + [{'type': 'text', 'text': task_instr}]
                messages = [{"role": "user", "content": content}]
            elif eval_mode == "interleaved":
                traj_data = json.load(open(os.path.join(traj_dir, f"{row['traj_id']}.json")))
                content = []

                for sub_task, sub_traj in zip(traj_data['sub_tasks'], traj_data['sub_trajs']):
                    subgoal_str = f"Your task goal: {sub_task['task_desc']}. "
                    content.append({'type': 'text', 'text': subgoal_str})
                    low_start, low_end = sub_traj['low_pddl_idx']

                    if low_start == 0:
                        content.append({'type': 'text', 'text': " State: "})
                        #content.append({'type': 'image', 'image': img_list[0]})
                        content.append({'type': 'image'})
                        ctx_img_list.append(img_list[0])
                    else:
                        content.append({'type': 'text', 'text': " State: "})
                        #content.append({'type': 'image', 'image': lowidx2imgs[low_start-1][-1]})
                        content.append({'type': 'image'})
                        ctx_img_list.append(lowidx2imgs[low_start-1][-1])

                    #breakpoint()
                    for low_idx in range(low_start, low_end):
                        act_dict = traj_data['plan']['low_actions'][low_idx]['api_action']
                        act_str = nieh_utils.act_dict_to_str(act_dict)
                        content.append({'type': 'text', 'text': f" Action: {act_str}"})
                        if len(lowidx2imgs[low_idx]) > 0:
                            content.append({'type': 'text', 'text': " State: "})
                        for low_img in lowidx2imgs[low_idx]:
                            #content.append({'type': 'image', 'image': low_img})
                            content.append({'type': 'image'})
                            ctx_img_list.append(low_img)

                content.append({'type': 'text', 'text': task_instr})
                messages = [{"role": "user", "content": content}]
            elif eval_mode == "text_state":
                if os.path.exists(os.path.join(traj_dir, f"{row['traj_id']}.txt")):
                    traj_data = json.load(open(os.path.join(traj_dir, f"{row['traj_id']}.txt")))
                elif os.path.exists(os.path.join(traj_dir, f"{row['traj_id']}.json")):
                    traj_data = json.load(open(os.path.join(traj_dir, f"{row['traj_id']}.json")))
                else:
                    raise ValueError()
                content = []

                last_low_idx = len(traj_data['plan']['low_actions']) - 1
                state_summary = traj_data['state_summary'][str(last_low_idx)]

                content.append({'type': 'text', 'text': f" State: {state_summary}; Last agent's view: "})
                content.append({'type': 'image'})
                ctx_img_list.append(img_list[-1])
                content.append({'type': 'text', 'text': task_instr})
                messages = [{"role": "user", "content": content}]
            else:
                # Default behaviour: use the full trajectory
                ctx_img_list = img_list
                content = [{'type': 'image', 'image': im} for im in ctx_img_list] + [{'type': 'text', 'text': task_instr}]
                messages = [{"role": "user", "content": content}]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            logger.info(
                f"gt_idx: {row['gt_img_idx']}, full_traj: True, "
                f"mode: {eval_mode}, n_imgs: {len(ctx_img_list)}, "
                f"{prompt}"
            )

            if eval_mode == "clip_retrieval":
                logger.info(f"selected_indices: {selected_indices}")

            lm_response, ctx_len = generate(ctx_img_list, video_path, messages, processor, model, device)
            if lm_response:  # Only process if we got a valid response
                score = nieh_utils.get_score(lm_response, row['answer'])
                logger.info(
                    f"gt_idx: {row['gt_img_idx']}, full_traj: True, "
                    f"n_imgs: {len(ctx_img_list)}, "
                    f"score: {score}, lm_response: {lm_response}, ans: {row['answer']}"
                )
                total_match += score
                total_count += 1.0
                
                if isinstance(row['answer'][0], str) and not row['answer'][0].lower() in ["no", "yes"]:
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
            # for haystack building
            NiH_match = [0 for _ in target_depths]
            NiH_count = [0 for _ in target_depths] 

            for di, depth in enumerate(target_depths):
                ctx_img_list, _ = nieh_utils.build_haystack(ctx_size, depth, row['gt_img_idx'], n_img_token, img_list)
                
                if len(ctx_img_list) > 0:
                    lm_response = generate(ctx_img_list, prompt, processor, model, device)
                    if lm_response:  # Only process if we got a valid response
                        score = nieh_utils.get_score(lm_response, row['answer'])
                        logger.info(
                            f"gt_idx: {row['gt_img_idx']}, ctx_size: {ctx_size}K, "
                            f"depth: {depth_map[depth]}, n_imgs: {len(ctx_img_list)}, "
                            f"score: {score}, lm_response: {lm_response}, ans: {row['answer']}"
                        )
                        NiH_match[di] = score
                        NiH_count[di] = 1.0

                        per_depth_results.append(
                            {
                                "depth": depth,
                                "depth_label": depth_map[depth],
                                "n_imgs": len(ctx_img_list),
                                "ctx_n_tokens": ctx_len,
                                "llm_response": lm_response,
                                "gt_answer": row.get("answer"),
                                "score": float(score),
                                "count": 1.0,
                            }
                        )

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
    
    logger.info("\nFinal Results:")

    if full_traj:
        if total_count > 0:
            score = total_match / total_count
            logger.info(
                f"score: {score:.4f}, total_match: {total_match}, "
                f"total_count: {total_count}"
            )
            logger.info(
                f"open_ended_score: {score:.4f}, open_ended_match: {oe_match}, "
                f"open_ended_count: {oe_count}"
            )
    else:
        for depth in target_depths:
            if total_count[depth] > 0:
                score = np.array(total_match[depth]) / np.array(total_count[depth])
                logger.info(
                    f"ctx_size: {ctx_size}K, depth: {depth_map[depth]}, "
                    f"score: {score:.4f}, total_match: {total_match[depth]}, "
                    f"total_count: {total_count[depth]}"
                )

    log_f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="model name",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=32,
        help="Context size in K tokens"
    )
    parser.add_argument(
        "--ctx_extension",
        type=str,
        help="Type of context length extension (e.g., 'longrope')"
    )
    parser.add_argument(
        "--ctx_extension_factor",
        type=float,
        default=4.0,
        help="Factor for context length extension"
    )
    parser.add_argument(
        "--qa_file_path",
        type=str,
        required=True,
        help="Path to the QA data file"
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        required=True,
        help="Directory containing metadata and images"
    )
    parser.add_argument(
        "--traj_dir",
        type=str,
        required=True,
        help="Directory containing trajectory files"
    )
    parser.add_argument(
        "--full_traj",
        action="store_true",
        help="Use the entire trajectory image list instead of building haystack"
    )
    parser.add_argument(
        "--n_img_token",
        type=int,
        default=None,
        help="Override per-image token count used in haystack sizing."
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"]
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="base model name for local checkpoints",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="full_traj",
        choices=["full_traj", "interleaved", "clip_retrieval", "truncate_head", "text_state", "video"],
        help=(
            "Evaluation mode when --full_traj is enabled. "
            "'full_traj': use entire trajectory; "
            "'clip_retrieval': retrieve top-K images with CLIP; "
            "'truncate_head': keep only the tail that fits in ctx_size."
        ),
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default=None,
        help="Optional CLIP model name for retrieval (only used when eval_mode='clip_retrieval'). "
             "Defaults to 'openai/clip-vit-large-patch14' if not set.",
    )
    parser.add_argument(
        "--clip_top_k",
        type=int,
        default=10,
        help="Number of top images to retrieve with CLIP when eval_mode='clip_retrieval'.",
    )


    args = parser.parse_args()

    if not os.path.exists(args.qa_file_path):
        raise FileNotFoundError(f"QA file path {args.qa_file_path} does not exist.")
    
    qa_data = []
    with open(args.qa_file_path, 'r') as f:
        data = csv.DictReader(f)
        for row in data:
            row_dict = dict(row)
            if 'gt_img_idx' in row_dict:
                row_dict['gt_img_idx'] = eval(row_dict['gt_img_idx'])
            if 'answer' in row_dict:
                row_dict['answer'] = eval(row_dict['answer'])
            qa_data.append(row_dict)
    
    main(
        qa_data=qa_data,
        traj_dir=args.traj_dir,
        metadata_dir=args.metadata_dir,
        model_name_or_path=args.model_name_or_path,
        ctx_size=args.ctx_size,
        ctx_extension=args.ctx_extension,
        ctx_extension_factor=args.ctx_extension_factor,
        full_traj=args.full_traj,
        attn_impl=args.attn_impl,
        base_model=args.base_model,
        qa_file_path=args.qa_file_path,
        eval_mode=args.eval_mode,
        clip_model_name=args.clip_model_name,
        clip_top_k=args.clip_top_k,
    )
