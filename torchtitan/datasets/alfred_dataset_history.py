import os
import re
import json
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

        # if not self.eval:
        #     self.max_seq_len = 131072
        
        self.split = split

        self.act_template = {
            "RotateLeft": "RotateLeft",
            "RotateRight": "RotateRight",
            "MoveAhead": "MoveAhead",
            "LookUp": "LookUp",
            "LookDown": "LookDown",
            "OpenObject": "OpenObject [object]",
            "CloseObject": "CloseObject [object]",
            "PickupObject": "PickupObject [object]",
            "PutObject": "PutObject [object] [receptacle]",
            "ToggleObjectOn": "ToggleObjectOn [object]",
            "ToggleObjectOff": "ToggleObjectOff [object]",
            "SliceObject": "SliceObject [object]",
            "NoOp": "NoOp",
        }

        self.traj_data_dir = traj_data_dir
        self.img_data_dir = img_data_dir
        self.traj_data = []
        self.data = []

        # Variables for checkpointing
        self._traj_idx = 0
        self._sample_idx = 0

        self.use_only_last_frame = True

        # self.system_prompt = (
        #     "You are an embodied AI agent operating in a simulated 3D environment. "
        #     "Perceive the scene (image inputs), and predict the next action to complete the task."
        # )
        self.system_prompt = (
            "You are an embodied AI agent operating in a simulated 3D environment. "
            "Your task is to perceive the scene from image inputs and predict the next action to complete the task.\n\n"
            
            "Available actions: RotateLeft, RotateRight, MoveAhead, LookUp, LookDown, OpenObject, "
            "CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject.\n\n"
            
            "Action constraints based on current state (last image):\n"
            "- Navigation (RotateLeft/RotateRight/MoveAhead): Only perform when safe and appropriate. "
            "If an object blocks your path, you cannot MoveAhead. When facing a wall, use RotateLeft or RotateRight to find another route.\n"
            "- PickupObject: Only valid when a target object is visible in your current view.\n"
            "- PutObject: Only valid when you are currently holding an object.\n"
            "- OpenObject/CloseObject: Only valid for openable objects (Cabinet, Fridge, Drawer, etc.).\n"
            "- SliceObject: Only valid when you are holding a ButterKnife. You must find and pick up the ButterKnife first before slicing.\n\n"
            
            "Given the history of previous actions and the current state, predict the next action to complete the task."
        )
        
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

            img_tar_file = filename.replace("txt", "tar") if "txt" in filename else filename.replace("json", "tar")
            tar_file = os.path.join(self.img_data_dir, img_tar_file)
            if os.path.exists(tar_file):
                img_dict = extract_and_convert_tar(tar_file, self.img_width, self.img_height)
            elif os.path.isdir(os.path.join(os.path.join(self.img_data_dir, img_tar_file.split(".")[0]))):
                img_dir = os.path.join(os.path.join(self.img_data_dir, img_tar_file.split(".")[0]))
                img_dict = read_images(img_dir, self.img_width, self.img_height)
            else:
                self._traj_idx = 0
                continue

            lowidx2img = defaultdict(list)
            for img_meta in traj['images']:
                lowidx2img[img_meta['low_idx']].append(img_meta['image_name'])

            N = len(traj['retrieved_image'])
            usable = (N // dp_world) * dp_world

            for si, (low_idx, history) in enumerate(traj['history_summary'].items()):
                if si >= usable:
                    break
                
                self._sample_idx += 1

                # Keep only trajectories owned by this shard
                if (si % dp_world) != dp_rank:
                    continue
                
                low_act = traj['plan']['low_actions'][int(low_idx)]['api_action']

                content, assistant_response, img_list = self._load_sample(int(low_idx), history, low_act, img_dict, lowidx2img)

                messages = [
                    {"role": "system", "content": self.system_prompt},
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
                # print(f"[rank{self.rank}][dp_rank{self.dp_rank}] traj_idx: {self._traj_idx} sample_idx: {self._sample_idx} input_ids: {output.input_ids.shape} n_img: {len(img_list)} prompt:\n{prompt}")
                # print(f"[rank{self.rank}][dp_rank{self.dp_rank}] labels: {self.processor.tokenizer.decode(labels[0, assistant_start_idx:]).strip()}")
                
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

    def get_obj_name(self, obj_id):
        return obj_id.split("|")[0]

    def act_dict_to_str(self, va):
        if va['action'] == 'PutObject':
            return f"PutObject {self.get_obj_name(va['objectId'])} {self.get_obj_name(va['receptacleObjectId'])}"
        elif 'Object' in va['action']:
            return f"{va['action']} {self.get_obj_name(va['objectId'])}"
        else:
            return va['action']

    def _load_sample(self, low_idx, history, low_act, img_dict, lowidx2img):
        contents = []
        imgs = []

        if low_idx == 0:
            contents.append({"type": "text", "text": f"Initial state: "})
            init_img = "000000000.jpg" if "000000000.jpg" in img_dict else "000000000.png"
            contents.append({"type": "image", "image": img_dict[init_img]})
            imgs.append(img_dict[init_img])
            contents.append({"type": "text", "text": f" Next action: "})
            assistant_response = f"{self.act_dict_to_str(low_act)}"
            return contents, assistant_response, imgs
        
        if history:
            contents.append({"type": "text", "text": f"Action history: {history};\n"})
        else:
            contents.append({"type": "text", "text": f"Action history: No history is available.\n"})

        contents.append({"type": "text", "text": f"Current state: "})

        # get current state (last low_idx's last image)
        if len(lowidx2img[low_idx-1]) > 0:
            cur_state_img = lowidx2img[low_idx-1][-1]
            if 'png' in cur_state_img:
                cur_state_img = cur_state_img.replace("png", "jpg")

            contents.append({"type": "image", "image": img_dict[cur_state_img]})
            imgs.append(img_dict[cur_state_img])

        contents.append({"type": "text", "text": f" Next action: "})
        assistant_response = f"{self.act_dict_to_str(low_act)}"
        
        return contents, assistant_response, imgs

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

    def _load_traj_data(self):
        directory_path = self.traj_data_dir
        if not os.path.exists(directory_path):
            raise ValueError(f"Trajectory data directory not found: {self.traj_data_dir}")
        
        all_files = [
            (str(file_path), file_path.name) 
            for file_path in Path(directory_path).rglob('*.txt')
        ]
        
        # Sort the file paths to ensure consistent order
        all_files.sort(key=lambda x: x[1]) # Sort by filename
        
        for file_path, file in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.traj_data.append({'text': f.read(), 'filename': file})
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from {file_path}: {str(e)}")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
    
    def get_templated_high_pddl_desc(self, high_pddl):
        a_type = high_pddl['discrete_action']['action']
        args = high_pddl['discrete_action']['args'] if 'args' in high_pddl['discrete_action'] else None

        if 'objectId' in high_pddl['planner_action']:
            objectId = high_pddl['planner_action']['objectId']
            obj_name = objectId.split("|")[0]
        if 'receptacleObjectId' in high_pddl['planner_action']:
            receptacleObjectId = high_pddl['planner_action']['receptacleObjectId']
            recep_name = receptacleObjectId.split("|")[0]

        templated_str = ""

        if 'GotoLocation' in a_type:
            templated_str = f"go to the {args[0]}"
        elif 'OpenObject' in a_type:
            templated_str = f"open the {obj_name}"
        elif 'CloseObject' in a_type:
            templated_str = f"close the {obj_name}"
        elif 'PickupObject' in a_type:
            templated_str = f"pick up the {obj_name}"
        elif 'PutObject' in a_type:
            templated_str = f"put the {obj_name} in the {recep_name}"
        elif 'CleanObject' in a_type:
            templated_str = f"wash the {obj_name}"
        elif 'HeatObject' in a_type:
            templated_str = f"heat the {obj_name}"
        elif 'CoolObject' in a_type:
            templated_str = f"cool the {obj_name}"
        elif 'ToggleObject' in a_type:
            templated_str = f"toggle {obj_name}"
        elif 'SliceObject' in a_type:
            templated_str = f"slice the {obj_name}"
        elif 'End' in a_type:
            templated_str = "<<STOP>>"

        return templated_str


class AlfredValidActDataLoader(ParallelAwareDataloader):
    def __init__(self, hf_ds: IterableDataset, dp_rank: int, dp_world_size: int,
                 batch_size: int,
                 eos_tok_id: int,
                 ignore_index: int = -100,
                 pin_memory: bool = True):
        # ds = BucketByLen(hf_ds)
        super().__init__(hf_ds, dp_rank, dp_world_size, batch_size, collate_fn=partial(self.collate_fn, eos_id=eos_tok_id))
        self.eos_tok_id = eos_tok_id
        self.ignore_index = ignore_index

    @staticmethod
    def _pad_1d(x: torch.Tensor, target_len: int, pad_id: int) -> torch.Tensor:
        # x shape: [1, L]
        need = target_len - x.shape[1]
        if need <= 0:
            return x
        pad = torch.full((x.shape[0], need), pad_id, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    @staticmethod
    def collate_fn(batch, eos_id, ignore_index=-100):
        # Unwrap if dataset already bundled the batch (BucketByLen behavior)
        if len(batch) == 1 and isinstance(batch[0], (list, tuple)) and batch[0] and isinstance(batch[0][0], dict):
            samples = batch[0]
        else:
            samples = batch

        # Pad text to max length
        max_len = max(s['input_ids'].shape[1] for s in samples)
        if max_len % 4 != 0: # for TP
            max_len += (4 - (max_len % 4))

        input_ids_list: List[torch.Tensor] = []
        labels_list:    List[torch.Tensor] = []

        # Per-sample image containers (do NOT stack)
        pixel_values_list: List[Optional[torch.Tensor]] = []
        image_grid_list:   List[Optional[torch.Tensor]] = []

        for s in samples:
            input_ids_list.append(
                AlfredValidActDataLoader._pad_1d(s['input_ids'], max_len, eos_id)
            )
            if 'labels' in s and s['labels'] is not None:
                labels_list.append(
                    AlfredValidActDataLoader._pad_1d(s['labels'], max_len, ignore_index)
                )

            # Keep images per-sample (Option A): pass through as-is
            pv = s.get('pixel_values', None)
            gh = s.get('image_grid_thw', None)

            # Allow “no image” samples; embed() will skip vision if no image tokens
            if isinstance(pv, torch.Tensor) and pv.numel() > 0:
                pixel_values_list.append(pv)
                image_grid_list.append(gh)
            else:
                pixel_values_list.append(None)
                image_grid_list.append(None)

        batch_dict = {
            'input_ids': torch.cat(input_ids_list, dim=0),  # [B, T]
            'pixel_values': pixel_values_list,               # List[Tensor|None], len B
            'image_grid_thw': image_grid_list,               # List[Tensor|None], len B
        }
        if len(labels_list):
            batch_dict['labels'] = torch.cat(labels_list, dim=0)  # [B, T]

        return batch_dict
