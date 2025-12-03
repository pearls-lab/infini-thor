import json
import os
import sys
import re
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import subprocess
from PIL import Image
from pathlib import Path

import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoProcessor,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration
)
from accelerate import init_empty_weights

import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def init_buffers(model, buffer_device, buffers_dict=None):
    # following the protocol of the original torchtitan repo,
    # but only init buffers here.
    # All parameters and buffers are empty after model.to_empty() called
    # Restore buffers after loading checkpoint
    
    # self: Qwen2_5_VLForActionPrediction
    # Qwen2_5_VLForConditionalGeneration
    # - self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
    # - self.model = Qwen2_5_VLModel(config)

    # Qwen2_5_VisionTransformerPretrainedModel
    # - self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    n_buffer_key = 0
    with torch.device(buffer_device):
        head_dim = model.visual.config.hidden_size // model.visual.config.num_heads
        dim, theta = head_dim // 2, 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        model.visual.rotary_pos_emb.inv_freq = inv_freq
        n_buffer_key += 1
        # this model.rotary_emb is for position embeddings
        inv_freq, _ = model.model.rotary_emb.rope_init_fn(model.model.rotary_emb.config, buffer_device)
        model.model.rotary_emb.inv_freq = inv_freq        
        n_buffer_key += 1
        # these are for rotary_emb in Attentions ...
        # NOTE: VisionTransformer layers don't have rotary_embs
        for layer in model.model.layers: # nn.ModuleList
            # type(layer) = Qwen2_5_VLDecoderLayer
            layer.self_attn.rotary_emb.inv_freq, _ = layer.self_attn.rotary_emb.rope_init_fn(
                layer.self_attn.rotary_emb.config, buffer_device
            )
            n_buffer_key += 1

    assert n_buffer_key == len(buffers_dict.keys()), f"Expected buffer keys: {buffers_dict.keys()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="HF model path or local directory with checkpoints")
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Constructing base model {args.base_model} and loading distributed checkpoint from {args.model_name_or_path} ...")

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)
        buffers_dict = {k: v.clone() for k, v in model.named_buffers()}

    model.to_empty(device="cpu")
    
    # Load distributed checkpoint into the model's state_dict
    # Issue: state_dict.keys() are not compaible in higher transformer verisions > 4.49.0
    state = {"model": model.state_dict()} # set(model.state_dict().keys()) == set(dict(model.named_parameters()).keys())
    print(f"Loading distributed checkpoint from: {args.model_name_or_path}")
    dcp.load(state, checkpoint_id=args.model_name_or_path)  # <-- NEW: distributed checkpoint load
    print("Checkpoint loaded successfully.")

    buffer_device = torch.device("cpu")
    init_buffers(model, buffer_device, buffers_dict)
    print("Buffers initialized.")

    buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
    print(f"[buffer] visual.rotary_pos_emb.inv_freq: {buffers_dict['visual.rotary_pos_emb.inv_freq']}")
    print(f"[buffer] model.layers.0.self_attn.rotary_emb.inv_freq: {buffers_dict['model.layers.0.self_attn.rotary_emb.inv_freq']}")

    # TODO: save model compatible with HF's classes.
    # 4. Materialize the model onto CPU
    # This is the crucial step: it moves the model from "meta" device to
    # a real device ("cpu"), allocating memory and applying the loaded weights.
    print("Materializing model onto CPU...")
    model.to(buffer_device)
    print("Model materialized.")

    # 5. Save the model and processor in standard HF format
    # We use safe_serialization=True to save as .safetensors
    print(f"Saving HF-compatible model and processor to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    
    print("\n--- Conversion Complete ---")
    print(f"Model saved to: {args.output_dir}")
    
'''
torchrun --nproc-per-node 1 convert_dcp_to_hf.py \
    --model_name_or_path /checkpoints/alfred-ft/step-9861/ \
    --base_model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir /checkpoints/alfred-ft/step-9861-hf/
'''
if __name__ == "__main__":

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)

    main()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()