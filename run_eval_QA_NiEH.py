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
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration

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

model2num_img_token = {
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': 1485, # base_image_feature (27, 27) + image_feature (+ new line) (27, 28)
}

depth_map = {
    0: 0,
    1: 0.2,
    2: 0.4,
    3: 0.6,
    4: 0.8,
}


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
    model_name: str,
    ctx_size: int = 32,
    target_depths: List[int] = [0, 1, 2, 3, 4],
    ctx_extension: Optional[str] = None,
    ctx_extension_factor: Optional[float] = None,
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Tokenizer setup
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.model_max_length = 1048576

    llm_config = AutoConfig.from_pretrained(model_name)
    
    if ctx_extension:
        logger.info(f"Using dynamic context length: {ctx_extension}")
        if ctx_extension == "longrope":
            llm_config.text_config.rope_scaling = {
                "rope_type": ctx_extension,
                "long_factor": ctx_extension_factor,
                "short_factor": 1,
                "factor": 1.0,
                "original_max_position_embeddings": llm_config.text_config.max_position_embeddings,
            }
        else:
            llm_config.text_config.rope_scaling = {
                "rope_type": ctx_extension,
                "factor": ctx_extension_factor,
                "original_max_position_embeddings": llm_config.text_config.max_position_embeddings,
            }
    
    logger.info("Loading model...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        device_map=device,
        low_cpu_mem_usage=True,
        config=llm_config)

    model.eval()

    n_img_token = model2num_img_token[model_name]

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

        NiH_match = [0 for _ in target_depths]
        NiH_count = [0 for _ in target_depths] 

        for di, depth in enumerate(target_depths):
            ctx_img_list = nieh_utils.build_haystack(ctx_size, depth, row['gt_img_idx'], n_img_token, img_list)
            
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
                    NiH_count[di] = 1

        total_match = [x + y for x, y in zip(total_match, NiH_match)]
        total_count = [x + y for x, y in zip(total_count, NiH_count)]
    
    # Print final results
    logger.info("\nFinal Results:")
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
        "--model_name",
        type=str,
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="model name",
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
        model_name=args.model_name,
        ctx_extension=args.ctx_extension,
        ctx_extension_factor=args.ctx_extension_factor,
    )
