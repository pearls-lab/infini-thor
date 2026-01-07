import os
import re
import json
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import itertools
from functools import partial
import random
import hashlib

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.tools.logging import logger
from torchtitan.datasets import ParallelAwareDataloader

from datasets import Dataset, load_dataset

from PIL import Image
import tarfile
from io import BytesIO


def extract_and_convert_tar(tar_path, img_width, img_height):
    """Extracts a .tar file and converts all .jpg files inside to a dictionary where keys are filenames and values are PIL images."""
    image_dict = {}
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and (member.name.lower().endswith(".jpg") or member.name.lower().endswith(".png")):
                file_obj = tar.extractfile(member)
                if file_obj:
                    base_filename = os.path.basename(member.name)
                    image = Image.open(BytesIO(file_obj.read()))
                    image = image.convert("RGB")  # Ensure consistent format
                    if image.size != (img_width, img_height):
                        image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
                    image_dict[base_filename] = image
    
    return image_dict

def pad_to_multiple(tensor, multiple=4, pad_token=0):
    length = tensor.shape[1]
    pad_length = (multiple - (length % multiple)) % multiple
    if pad_length > 0:
        pad_tensor = torch.full((tensor.shape[0], pad_length), pad_token, dtype=tensor.dtype)
        # NOTE pad in the head -- for the consistency with inference with CP
        tensor = torch.cat([pad_tensor, tensor], dim=1)
    return tensor


def pad_to_max_seq(tensor, max_seq=8192, pad_token=0):
    length = tensor.shape[1]
    pad_length = max_seq - length
    if pad_length > 0:
        pad_tensor = torch.full((tensor.shape[0], pad_length), pad_token, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
    return tensor


def _pin_batch(obj):
    if isinstance(obj, torch.Tensor):
        # Only CPU tensors can be pinned
        return obj if obj.is_cuda else obj.pin_memory()
    if isinstance(obj, dict):
        return {k: _pin_batch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_pin_batch(v) for v in obj)
    return obj


class ALFREDDataset(IterableDataset, Stateful):

    def __init__(
        self,
        dataset_name: str,
        processor,
        n_tok_per_img: int,
        img_width: int,
        img_height: int,
        img_token_id: int = None,
        traj_data_dir: str = "",
        img_data_dir: str = "",
        split: str = "train",
        max_seq_len: int = 131072,
        #world_size: int = 1,
        pad_to: int = 1,
        rank: int = 0,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        ignore_index: int = -100,
        eval: bool = False
    ) -> None:
        self.dataset_name = dataset_name
       
        self.processor = processor
        self.n_tok_per_img = n_tok_per_img
        self.img_width = img_width
        self.img_height = img_height
        self.max_seq_len = max_seq_len
        self.infinite = infinite
        self.img_tok_id = img_token_id if img_token_id else processor.tokenizer('<image>').input_ids[0]
        self.img_token = processor.tokenizer.decode([self.img_tok_id])
        self.act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
        self.eos_tok_id = processor.tokenizer.eos_token_id
        self.pad_tok_id = processor.tokenizer.pad_token_id
        self.ignore_index = ignore_index
        self.eval = eval
        #self.world_size = world_size
        self.pad_to = pad_to
        self.rank = rank
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        
        self.split = split

        self.traj_data_dir = traj_data_dir
        self.img_data_dir = img_data_dir
        self.traj_data = []
        self.data = []

        # Variables for checkpointing
        self._traj_idx = 0
        self._sample_idx = 0

        self.use_only_last_frame = True
        
        if len(self.data) == 0:
            self._load_data()

    def __len__(self):
        return len(self.traj_data)

    def _get_data_iter(self):
        it = iter(self.data)
        for _ in range(self._traj_idx): # iterator starting at sample_idx (if sample_idx is not 0 from the dataloader state)
            next(it)
        return it

    def __iter__(self):
        # Per-rank sharding
        dp_rank = self.dp_rank
        dp_world = max(1, self.dp_world_size)

        # Reset if we've completed an epoch
        if self._traj_idx >= len(self.data):
            self._traj_idx = 0
            self._sample_idx = 0

        it = self._get_data_iter()

        # Resume offsets
        start_traj = self._traj_idx
        start_sample = self._sample_idx
        
        # Iterate trajectories; select only those belonging to this shard
        for ti, (traj, filename) in enumerate(it, start=start_traj):
            self._traj_idx = ti + 1
            
            traj_id = traj['traj_id']

            img_tar_file = f"{traj_id}.tar"
            tar_file = os.path.join(self.img_data_dir, img_tar_file)
            
            if os.path.exists(tar_file):
                img_dict = extract_and_convert_tar(tar_file, self.img_width, self.img_height)
            elif os.path.isdir(os.path.join(os.path.join(self.img_data_dir, img_tar_file.split(".")[0]))):
                img_dir = os.path.join(os.path.join(self.img_data_dir, img_tar_file.split(".")[0]))
                img_dict = read_images(img_dir, self.img_width, self.img_height)
            else:
                self._traj_idx = 0
                continue

            N = len(traj['QA'])
            usable = (N // dp_world) * dp_world

            for si, (entry) in enumerate(traj['QA']):
                # entry has `question`, `answer`
                if si >= usable:
                    break
                
                self._sample_idx += 1

                # Keep only trajectories owned by this shard
                if (si % dp_world) != dp_rank:
                    continue

                content, img_list = self._load_sample(entry['question'], entry['qtype'], img_dict)

                assistant_response = entry['answer'][0] if isinstance(entry['answer'], list) else entry['answer']
                if not isinstance(assistant_response, str):
                    assistant_response = str(assistant_response)
                
                messages = [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": assistant_response}
                ]
                
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,   # set True if you plan to .generate immediately
                )

                output = self.processor(text=prompt, images=img_list, return_tensors="pt")

                # Tokenize assistant response (without special tokens)
                assistant_tokens = self.processor.tokenizer(
                    assistant_response, 
                    add_special_tokens=False
                ).input_ids
                
                # Search for assistant tokens in the actual output sequence
                seq = output.input_ids[0].tolist()
                assistant_start_idx = None
                alen = len(assistant_tokens)
                
                # Search from the end since assistant response is at the end
                for i in range(len(seq) - alen, -1, -1):
                    if seq[i:i+alen] == assistant_tokens:
                        assistant_start_idx = i
                        break
                
                if assistant_start_idx is None:
                    logger.warning(f"Could not find exact assistant token match")
                    assistant_start_idx = 0   

                labels = output.input_ids.clone()
                labels[:] = self.ignore_index
                labels[0, assistant_start_idx:] = output.input_ids[0, assistant_start_idx:]

                shift_input_ids = output.input_ids[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                shift_input_ids = pad_to_multiple(shift_input_ids, self.pad_to, pad_token=self.pad_tok_id)
                shift_labels = pad_to_multiple(shift_labels, self.pad_to, pad_token=self.ignore_index)

                logger.info(f"[rank{self.rank}][dp_rank{self.dp_rank}] traj_idx: {self._traj_idx} sample_idx: {self._sample_idx} input_ids: {output.input_ids.shape} n_img: {len(img_list)} prompt:\n{prompt}")
                logger.info(f"[rank{self.rank}][dp_rank{self.dp_rank}] labels: {self.processor.tokenizer.decode(labels[0, assistant_start_idx:]).strip()}")
                print(f"[rank{self.rank}][dp_rank{self.dp_rank}] traj_idx: {self._traj_idx} sample_idx: {self._sample_idx} input_ids: {output.input_ids.shape} n_img: {len(img_list)} prompt:\n{prompt}")
                print(f"[rank{self.rank}][dp_rank{self.dp_rank}] labels: {self.processor.tokenizer.decode(labels[0, assistant_start_idx:]).strip()}")
                
                yield {
                    'input_ids': shift_input_ids,
                    'pixel_values': output.pixel_values,
                    'labels': shift_labels,
                    'image_grid_thw': output.image_grid_thw
                }
            # end of one traj
            
        # end of epoch
        self._sample_idx = 0
        self._traj_idx = len(self.traj_data)


    def load_state_dict(self, state_dict):
        logger.info(f"loading Dataloader state_dict ... : {state_dict}")
        self._sample_idx = state_dict['sample_idx']
        self._traj_idx = state_dict['traj_idx']

    def state_dict(self):
        return {"sample_idx": self._sample_idx, "traj_idx": self._traj_idx}

    def _load_sample(self, question, question_type, img_dict, num_samples=32):
        '''
        Following VSI-Bench's input format:
            [VideoFrames][Pre-prompt][Question][Post-prompt]
            
            Pre-prompt: "These are frames of a video."
            Post-prompt:
                NA: "Please answer the question using a single word or phrase."
                MCA: "Answer with the option's letter from the given choices directly."
                
        img_dict: img_dict["000000000.png"] = PIL image instance
        '''
        contents = []
        imgs = []
        
        img_pool = sorted(img_dict.keys())  # filenames
        n = len(img_pool)
        k = min(num_samples, n)  # can't sample more unique frames than available
        # choose k indices uniformly from [0, n-1]
        idxs = np.linspace(0, n - 1, num=k)
        idxs = np.round(idxs).astype(int)

        # optional: ensure uniqueness while keeping order (linspace rounding can duplicate when n<k, but we clamp k<=n)
        idxs = np.unique(idxs)

        for i in idxs:
            key = img_pool[i]
            img = img_dict[key]
            contents.append({"type": "image", "image": img})
            imgs.append(img)
        
        # pre-prompt
        contents.append({"type": "text", "text": f"These are frames of a video.\n"})
        # question
        contents.append({"type": "text", "text": f"{question}\n"})
        
        # post-prompt
        if question_type in ["relative_direction", "relative_distance"]:
            contents.append({"type": "text", "text": f"Answer with the option's letter from the given choices directly."})
        else:
            contents.append({"type": "text", "text": f"Please answer the question using a single word or phrase."})
        
        return contents, imgs

    def _load_data(self):
        directory_path = self.traj_data_dir
        if not os.path.exists(directory_path):
            raise ValueError(f"Trajectory data directory not found: {self.traj_data_dir}")
        
        all_files = [
            (str(file_path), file_path.name) 
            for file_path in list(Path(directory_path).rglob('*.txt')) + 
                            list(Path(directory_path).rglob('*.json'))
        ]

        if len(all_files) == 0:
            raise ValueError(f"Files not found at : {self.traj_data_dir}")
        
        # Sort the file paths to ensure consistent order
        all_files.sort(key=lambda x: x[1]) # Sort by filename
        
        for file_path, file in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    traj = json.loads(f.read())
                    self.data.append((traj, file))
                    self.traj_data.append(traj)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from {file_path}: {str(e)}")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")


class AlfredDataLoader(ParallelAwareDataloader):

    def __init__(self, 
        hf_ds: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        pin_memory: bool = True):
        super().__init__(hf_ds, dp_rank, dp_world_size, batch_size, collate_fn=self.collate_fn)    

    @staticmethod
    def collate_fn(batch):
        max_img_len = max(sample['pixel_values'].size(0) for sample in batch)
        
        input_ids = []
        pixel_values = []
        n_image = []
        labels = []
        image_grid_thw = []
        
        for bi, sample in enumerate(batch):
            input_ids.append(sample['input_ids'])

            pad_len = max_img_len - sample['pixel_values'].size(0)

            if pad_len > 0:
                pad_shape = (pad_len, *sample['pixel_values'].shape[1:])
                # IMPORTANT: keep on CPU here; pinning happens after collate
                # padding = torch.zeros(pad_shape, dtype=sample['pixel_values'].dtype, 
                #                 device=sample['pixel_values'].device)
                padding = torch.zeros(pad_shape, dtype=sample['pixel_values'].dtype)
                pixel_values.append(torch.cat([sample['pixel_values'], padding], dim=0))
            else:
                pixel_values.append(sample['pixel_values'])
            
            n_image.append([sample['pixel_values'].shape[0]])
            labels.append(sample['labels'])
            image_grid_thw.append(sample['image_grid_thw'])

        # Keep everything on CPU; DataLoader (or our wrapper) will pin
        # TODO: visual pad mask ?
        batch_dict = {
            'input_ids': torch.concat(input_ids, dim=0),
            'pixel_values': torch.concat(pixel_values, dim=0),
            'n_image': torch.tensor(n_image, device=input_ids[0].device, dtype=input_ids[0].dtype),
            'image_grid_thw': torch.concat(image_grid_thw, dim=0),
        }
        
        if labels:
            batch_dict['labels'] = torch.concat(labels, dim=0)

        return batch_dict
