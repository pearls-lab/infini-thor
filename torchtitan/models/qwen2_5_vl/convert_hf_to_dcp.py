import os
import torch
from transformers import AutoConfig, AutoProcessor, AutoModel

import torch.distributed.checkpoint as dcp

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    #config=config
)

# TODO: Convert and save the distributed checkpoint
# Save the distributed checkpoint
output_dir = "./checkpoints"

#DCP.save({"model": model.state_dict()}, storage_writer=DCP.filesystem.FileSystemWriter(output_dir, thread_count=1))
#DCP.save(model.state_dict(), storage_writer=DCP.filesystem.FileSystemWriter(output_dir, thread_count=1))
dcp.save({"model": model.state_dict()}, checkpoint_id=output_dir)

print(f"Distributed checkpoint saved at {output_dir}")
