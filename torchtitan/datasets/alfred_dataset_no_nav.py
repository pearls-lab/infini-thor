import os
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import itertools

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
        cp_degree: int = 1,
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
        self.ignore_index = ignore_index
        self.eval = eval
        #self.world_size = world_size
        self.cp_degree = cp_degree
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

        # Variables for checkpointing
        self._sample_idx = 0
        self._chunk_idx = 0

        self.use_only_last_frame = True

        self.system_prompt = "You are an embodied AI agent operating in a simulated 3D environment. " + \
                            "Perceive the scene (image inputs), and predict the next action to complete the task."

        if len(self.traj_data) == 0:
            self._load_traj_data()

    def __len__(self):
        return len(self.traj_data)

    def _get_data_iter(self):
        if self._sample_idx >= len(self.traj_data): # reset 
            self._sample_idx = 0
            self._chunk_idx = 0

        it = iter(self.traj_data)
        for _ in range(self._sample_idx): # iterator starting at sample_idx (if sample_idx is not 0 from the dataloader state)
            next(it)
        return it

    def __iter__(self):

        # Per-rank sharding
        dp_rank = self.dp_rank
        dp_world = max(1, self.dp_world_size)

        N = len(self.traj_data)
        usable = (N // dp_world) * dp_world  # drop the tail so every rank has equal count

        # Resume offsets
        start_traj = self._sample_idx
        start_chunk = self._chunk_idx

        # Iterate trajectories; select only those belonging to this shard
        for ti, traj in enumerate(self._get_data_iter(), start=start_traj):
        #for ti, traj in enumerate(self.traj_data, start=start_traj): -> this doens't work when len(self.traj_data) % dp_world_size != 0
            # Stop exactly at the dropped tail boundary
            if ti >= usable:
                break

            # Always advance sample cursor so we can't get stuck if we skip
            self._sample_idx = ti + 1
            
            # Keep only trajectories owned by this shard
            if (ti % dp_world) != dp_rank:
                # if we skip a traj, and we were resuming inside it, reset chunk cursor
                if ti == start_traj:
                    self._chunk_idx = 0
                continue

            if self.eval:
                yield json.loads(traj['text'])
                self._sample_idx = ti + 1
                self._chunk_idx = 0
                continue

            filename = traj['filename']
            img_tar_file = filename.replace("txt", "tar")
            tar_file = os.path.join(self.img_data_dir, img_tar_file)
            if not os.path.exists(tar_file):
                self._chunk_idx = 0
                continue

            # Heavy work happens ONLY for this shard's trajectories
            content, img_list = self._load_sample(traj, chunk=True)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]

            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,   # set True if you plan to .generate immediately
            )

            output = self.processor(text=prompt, images=img_list, return_tensors="pt")

            #logger.info(f"[rank{self.rank}][dp_rank{self.dp_rank}] sample_idx: {self._sample_idx} n_img: {len(img_list)}\nprompt: {prompt}")
            print(f"[rank{self.rank}][dp_rank{self.dp_rank}] sample_idx: {self._sample_idx} n_img: {len(img_list)}\nprompt: {prompt}")
            
            labels = output.input_ids.clone()

            act_tok = False
            for i, l in enumerate(labels[0]):
                if (not act_tok) and l == self.act_tok_id: # 151648
                    act_tok = True
                    continue
                
                if (not act_tok) and l != self.act_tok_id:
                    labels[0][i] = self.ignore_index

                if act_tok and l == self.act_tok_id:
                    act_tok = False
            
            input_ids = output.input_ids[:, :-1]
            labels = labels[:, 1:]

            input_ids = pad_to_multiple(input_ids, self.cp_degree, pad_token=self.eos_tok_id)
            labels = pad_to_multiple(labels, self.cp_degree, pad_token=self.ignore_index)

            yield {
                'input_ids': input_ids,
                'pixel_values': output.pixel_values,
                'labels': labels,
                'image_grid_thw': output.image_grid_thw
            }
            # end of one traj
            
        # end of epoch
        self._sample_idx = len(self.traj_data)
        self._chunk_idx = 0

    def build_messages_from_interleaved(self, lang_input: str, img_list):
        """
        Turn:  text <|image_pad|> text <|image_pad|> ... text
        into:  [{"role":"user","content":[{"type":"text",...},{"type":"image"}, ... ]}]
        """
        parts = lang_input.split(self.img_token)
        # assert len(parts) - 1 == len(img_list), \
        #     f"#<|image_pad|> ({len(parts)-1}) must equal #images ({len(img_list)})"
        assert len(parts) - 1 == len(img_list), \
            f"#<|image_pad|> ({lang_input}) \n\n parts: ({parts})"
        
        content = []
        for i, chunk in enumerate(parts):
            if chunk:
                content.append({"type": "text", "text": chunk})
            if i < len(img_list):
                content.append({"type": "image"})  # image placeholder in order

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content}
        ]
        return messages

    def load_state_dict(self, state_dict):
        logger.info(f"loading Dataloader state_dict ... : {state_dict}")
        self._sample_idx = state_dict['sample_idx']
        self._chunk_idx = state_dict['chunk_idx']

    def state_dict(self):
        return {"sample_idx": self._sample_idx, "chunk_idx": self._chunk_idx}

    def _load_sample(self, traj, chunk=True):
        # with S3
        filename = traj['filename'] # with S3
        traj = json.loads(traj['text'])
        contents = self.seq_preprocess(traj)

        img_tar_file = filename.replace("txt", "tar")
        tar_file = os.path.join(self.img_data_dir, img_tar_file)

        img_dict = extract_and_convert_tar(tar_file, self.img_width, self.img_height)

        imgs = []
        
        for content in contents:
            if content["type"] == "image":
                content["image"] = img_dict[content["image"].replace("png", "jpg")]
                imgs.append(content["image"])
        
        return contents, imgs

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

    def seq_preprocess(self, traj):
        contents = []

        # Prepare: low_idx_to_image
        low_idx_2_image = defaultdict(list)
        for im_info in traj['images']:
            low_idx_2_image[im_info['low_idx']].append(im_info['image_name'])

        # Prepare
        high_idx_2_low_act_list = defaultdict(list)
        for low_idx, low_act in enumerate(traj['plan']['low_actions']):
            high_idx = low_act['high_idx']
            low_act['low_idx'] = low_idx
            if len(high_idx_2_low_act_list[high_idx]) > 0:
                assert high_idx_2_low_act_list[high_idx][-1]['low_idx'] < low_act['low_idx']
            high_idx_2_low_act_list[high_idx].append(low_act)

        # start: make squences here
        main_goal_str = "<|goal|>Your task goal: "
        if 'turk_annotations' in traj:
            main_goal_str += traj['turk_annotations']['anns'][0]['task_desc'] + "<|goal|>"
        # else we need to use templated desc .. later

        contents.append({"type": "text", "text": main_goal_str})
        contents.append({"type": "image", "image": '000000000.png'})

        for high_idx, low_act_list in high_idx_2_low_act_list.items():
            high_action = traj['plan']['high_pddl'][high_idx]['discrete_action']
            if high_action['action'] == "NoOp":
                continue

            if high_action['action'] == "GotoLocation":
                # if high_idx + 1 < len(traj['plan']['high_pddl']) and traj['plan']['high_pddl'][high_idx+1]['discrete_action']['action'] == "PickupObject":
                #     goto_loc = high_action['args'][0]
                #     pickup_obj = traj['plan']['high_pddl'][high_idx+1]['discrete_action']['args'][0]
                #     dest = goto_loc if goto_loc != pickup_obj else pickup_obj
                dest = high_action['args'][0]
                action_str = f"<|act|>GotoLocation {dest}<|act|>"
                contents.append({"type": "text", "text": action_str})
                for _, low_act in enumerate(low_act_list):
                    low_idx = low_act['low_idx']
                contents.append({"type": "image", "image": low_idx_2_image[low_idx][-1]})
            else:
                for _, low_act in enumerate(low_act_list):
                    low_idx = low_act['low_idx']
                    action_str = self.serialize_action(low_act['api_action'])
                    contents.append({"type": "text", "text": action_str})
                    contents.append({"type": "image", "image": low_idx_2_image[low_idx][-1]})

        return contents

    def serialize_action(self, act):
        template = self.act_template[act['action']]
        if 'objectId' in act:
            template = template.replace("[object]", act['objectId'].split("|")[0])
        if 'receptacleObjectId' in act:
            template = template.replace("[receptacle]", act['receptacleObjectId'].split("|")[0])
        return '<|act|>' + template + '<|act|>'
    
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
