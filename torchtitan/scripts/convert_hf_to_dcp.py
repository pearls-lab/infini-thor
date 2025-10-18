import argparse
import os
import torch
from transformers import AutoConfig, AutoProcessor, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration
import torch.distributed.checkpoint as dcp

modelname2class = {
    'Qwen/Qwen2.5-VL-7B-Instruct': Qwen2_5_VLForConditionalGeneration
}

def main():
    parser = argparse.ArgumentParser(description="Save distributed checkpoint for a model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name or local path (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the distributed checkpoint."
    )
    args = parser.parse_args()

    model_cls = modelname2class[args.model_name]
    model = model_cls.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dcp.save({"model": model.state_dict()}, checkpoint_id=args.output_dir)

    print(f"Distributed checkpoint saved at {args.output_dir}")

if __name__ == "__main__":
    main()