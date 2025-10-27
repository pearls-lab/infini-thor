import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
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
    Qwen2_5_VLForConditionalGeneration
)
from transformers.utils import is_flash_attn_2_available, is_torch_cuda_available

import utils.nieh_utils as nieh_utils

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
        )
        model = model_cls.from_pretrained(model_name_or_path, **kwargs)
        #model.to(device)

        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    model.eval()

    return model, cfg, processor


@torch.no_grad()
def generate(
    img_list: List[Image.Image], 
    prompt: str, 
    processor: Any, 
    model: Any, 
    device: torch.device
) -> str:
    content = []
    for im in img_list:
        content.append({'type': 'image'})
    
    content.append({'type': 'text', 'text': prompt})

    messages = [
        {"role": "user", "content": content}
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(images=img_list, text=prompt, padding=True, return_tensors="pt").to(device, torch.bfloat16)
    out = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.eos_token_id)
    
    decoded_out = processor.batch_decode(out, skip_special_tokens=True)
    lm_response = decoded_out[0].strip().split("\n")[-1]
    return lm_response


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
    base_model: str = None,
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

    if full_traj:
        total_match = 0.0
        total_count = 0.0
    else:
        total_match = [0 for _ in target_depths]
        total_count = [0 for _ in target_depths]

    for qidx, row in enumerate(qa_data):
        img_list, metadata, traj_text, img_path_list = nieh_utils.load_qa_data(row['traj_id'], metadata_dir)
        assert len(img_list) == len(metadata['img_idx']), f"Image count mismatch for trajectory {row['traj_id']}"

        prompt = f"""
        These images are the agent's view in time order. Answer the question given the images.
        Do not include explanation or reasoning in the answer. Answer with a single word or words.
        {row['question']}
        """

        if full_traj:
            ctx_img_list = img_list
            logger.info(
                    f"gt_idx: {row['gt_img_idx']}, full_traj: True, "
                    f"n_imgs: {len(ctx_img_list)}, ")

            lm_response = generate(ctx_img_list, prompt, processor, model, device)
            if lm_response:  # Only process if we got a valid response
                score = nieh_utils.get_score(lm_response, row['answer'])
                logger.info(
                    f"gt_idx: {row['gt_img_idx']}, full_traj: True, "
                    f"n_imgs: {len(ctx_img_list)}, "
                    f"score: {score}, lm_response: {lm_response}, ans: {row['answer']}"
                )
                total_match += score
                total_count += 1.0
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
    else:
        for depth in target_depths:
            if total_count[depth] > 0:
                score = np.array(total_match[depth]) / np.array(total_count[depth])
                logger.info(
                    f"ctx_size: {ctx_size}K, depth: {depth_map[depth]}, "
                    f"score: {score:.4f}, total_match: {total_match[depth]}, "
                    f"total_count: {total_count[depth]}"
                )

    
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
        metadata_dir=args.metadata_dir,
        model_name_or_path=args.model_name_or_path,
        ctx_size=args.ctx_size,
        ctx_extension=args.ctx_extension,
        ctx_extension_factor=args.ctx_extension_factor,
        full_traj=args.full_traj,
        attn_impl=args.attn_impl,
        base_model=args.base_model
    )
