import os
import re
import json
import random
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


class InfiniTHORDataset(IterableDataset, Stateful):

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
            "SliceObject": "SliceObject [object]"
        }

        self.all_actions = list(self.act_template.keys())

        self.traj_data_dir = traj_data_dir
        self.img_data_dir = img_data_dir
        self.traj_data = []

        # Variables for checkpointing
        self._sample_idx = 0
        self._chunk_idx = 0

        self.use_only_last_frame = True

        self.system_prompt = (
            "You are an embodied AI agent operating in a simulated 3D environment. "
            "Perceive the scene (image inputs), and predict the next action to complete the task."
        )
        
        if len(self.traj_data) == 0:
            self._load_traj_data()

    def __len__(self):
        return len(self.traj_data)

    def _get_data_iter(self):
        # Create iterator and skip to current position
        it = iter(self.traj_data)
        for _ in range(self._sample_idx): # iterator starting at sample_idx (if sample_idx is not 0 from the dataloader state)
            next(it)
        return it

    def __iter__(self):
        # for per-rank sharding
        dp_world = max(1, self.dp_world_size)

        N = len(self.traj_data)
        usable = (N // dp_world) * dp_world  # drop the tail so every rank has equal count

        # Reset if we've completed an epoch
        if self._sample_idx >= len(self.traj_data):
            self._sample_idx = 0
            self._chunk_idx = 0

        it = self._get_data_iter()

        # Resume offsets
        start_traj = self._sample_idx
        start_chunk = self._chunk_idx

        # Iterate trajectories; select only those belonging to this shard
        for ti, traj in enumerate(it, start=start_traj):
        #for ti, traj in enumerate(self.traj_data, start=start_traj): -> this doens't work when len(self.traj_data) % dp_world_size != 0
            # Stop exactly at the dropped tail boundary
            if ti >= usable:
                break

            # Always advance sample cursor so we can't get stuck if we skip
            self._sample_idx = ti + 1
            
            # Keep only trajectories owned by this shard
            if (ti % dp_world) != self.dp_rank:
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
            chunks = self._load_sample(traj)
            if not isinstance(chunks, list):
                chunks = [chunks]

            # Resume inside the first selected trajectory if needed
            first_chunk_idx = start_chunk if ti == start_traj else 0

            for ci, chunk in enumerate(chunks[first_chunk_idx:], start=first_chunk_idx):
                n_img_token = chunk['chosen_lang'].count(self.img_token)
                if not self.use_only_last_frame and len(chunk['img_list']) != n_img_token:
                    logger.warning(f"Image mismatch in chunk. Skipping.")
                    continue

                # --- Process Chosen ---
                chosen_messages = self.build_messages_from_interleaved(chunk['chosen_lang'], chunk['img_list'])
                chosen_prompt = self.processor.tokenizer.apply_chat_template(
                    chosen_messages, tokenize=False, add_generation_prompt=False
                )
                chosen_out = self.processor(
                    text=chosen_prompt, images=chunk['img_list'], return_tensors="pt"
                )
                
                # --- Process Rejected ---
                rejected_messages = self.build_messages_from_interleaved(chunk['rejected_lang'], chunk['img_list'])
                rejected_prompt = self.processor.tokenizer.apply_chat_template(
                    rejected_messages, tokenize=False, add_generation_prompt=False
                )
                rejected_out = self.processor(
                    text=rejected_prompt, images=chunk['img_list'], return_tensors="pt"
                )

                # --- Create Labels (Masking user/system tokens) ---
                c_labels = self._create_labels(chosen_out.input_ids)
                r_labels = self._create_labels(rejected_out.input_ids)

                # --- Pad & Yield ---
                c_input_ids = pad_to_multiple(chosen_out.input_ids[:, :-1], self.pad_to, self.eos_tok_id)
                c_labels = pad_to_multiple(c_labels[:, 1:], self.pad_to, self.ignore_index)
                
                r_input_ids = pad_to_multiple(rejected_out.input_ids[:, :-1], self.pad_to, self.eos_tok_id)
                r_labels = pad_to_multiple(r_labels[:, 1:], self.pad_to, self.ignore_index)

                self._chunk_idx = ci + 1

                yield {
                    'chosen_input_ids': c_input_ids,
                    'chosen_labels': c_labels,
                    'rejected_input_ids': r_input_ids,
                    'rejected_labels': r_labels,
                    'pixel_values': chosen_out.pixel_values, # Shared image tensors
                    'image_grid_thw': chosen_out.image_grid_thw, # Shared grid info
                    'n_image': len(chunk['img_list'])
                }

            # end of one traj
            # reset chunk_idx
            self._chunk_idx = 0
            
        # end of epoch
        self._sample_idx = len(self.traj_data)
        self._chunk_idx = 0

    def _create_labels(self, input_ids):
        """Standard label creation with masking for non-action tokens"""
        labels = input_ids.clone()
        act_tok = False
        for i, l in enumerate(labels[0]):
            if (not act_tok) and l == self.act_tok_id:
                act_tok = True
                continue
            if (not act_tok) and l != self.act_tok_id:
                labels[0][i] = self.ignore_index
            if act_tok and l == self.act_tok_id:
                act_tok = False
        return labels

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

    def _load_sample(self, traj_entry):
        """Loads a trajectory and generates Chosen/Rejected pairs."""
        filename = traj_entry['filename']
        traj = json.loads(traj_entry['text'])
        
        # Preprocess to get aligned sequences and images
        # We need a new preprocessor that returns chosen/rejected pairs
        chunk_data = self.seq_preprocess_dpo(traj)

        img_tar_file = filename.replace("txt", "tar")
        tar_file = os.path.join(self.img_data_dir, img_tar_file)
        img_dict = extract_and_convert_tar(tar_file, self.img_width, self.img_height)
        
        chunks = []
        
        # chunk_data contains list of (chosen_seq, rejected_seq, img_filenames)
        for c_seq, r_seq, c_imgs in chunk_data:
            # if self.dataset_name == "alfred":
            #     _img_list = [img_dict[fname.replace("png", "jpg")] for fname in c_imgs]
            # else:
            _img_list = [img_dict[fname] for fname in c_imgs]
                
            chunks.append({
                'chosen_lang': c_seq,
                'rejected_lang': r_seq,
                'img_list': _img_list,
            })

        return chunks

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

    def seq_preprocess_dpo(self, traj):
        """
        Generates paired (chosen, rejected) text sequences.
        Chosen = Ground Truth
        Rejected = GT Sequence but with the LAST action swapped for a random incorrect one.
        """
        low_idx_2_image = defaultdict(list)
        for im_info in traj['images']:
            low_idx_2_image[im_info['low_idx']].append(im_info['image_name'])

        # --- 1. Build Object Pool from Trajectory ---
        # We collect every object ID mentioned in the expert demonstration.
        # This ensures our negative samples reference objects that actually exist in the scene.
        scene_objects = set()
        if 'plan' in traj and 'low_actions' in traj['plan']:
            for act in traj['plan']['low_actions']:
                api_act = act['api_action']
                if 'objectId' in api_act:
                    scene_objects.add(api_act['objectId'])
                if 'receptacleObjectId' in api_act:
                    scene_objects.add(api_act['receptacleObjectId'])
        
        # Convert to list for random sampling
        scene_objects = list(scene_objects)

        chunk_output = [] # List of tuples (chosen_str, rejected_str, img_files_list)
        
        # Basic setup same as SFT
        n_system_prompt_tok = len(self.processor(text=self.system_prompt).input_ids) + 8 
        tok_buffer_size = n_system_prompt_tok

        # NOTE: For DPO in this specific recursive "Infini" format, 
        # we simplify: we generate pairs for each high-level plan segment.
        
        chunk_seq_base = "" # accumulated context
        n_chunk_tok = 0
        chunk_img_files = []
        last_state_image = '000000000.png'
        
        for sub_traj in traj['sub_trajs']:
            main_goal_str = f"<|goal|>Your task goal: {sub_traj['subgoal']}<|goal|>"
            chunk_seq_base += main_goal_str
            # Add initial image
            chunk_seq_base += self.img_token
            chunk_img_files.append(last_state_image)
            
            low_start, low_end = sub_traj['low_pddl_idx']
            for high_idx in range(*sub_traj['high_pddl_idx']):
                low_act_list = [act for act in traj['plan']['low_actions'][low_start:low_end] if act['high_idx'] == high_idx]
                
                # --- Construct Chosen Sequence for this Step ---
                high_plan_seq_chosen = ""
                step_imgs = []
                
                for _, low_act in enumerate(low_act_list):
                    action_str = self.serialize_action(low_act['api_action'])
                    low_idx = low_act['low_idx']
                    # Add image + action
                    # For simplicity, assuming last frame usage
                    high_plan_seq_chosen += (self.img_token + action_str)
                    step_imgs.append(low_idx_2_image[low_idx][-1])

                # --- Construct Rejected Sequence for this Step ---
                # We take the Chosen sequence, but change the LAST action.
                # This makes the rejection "hard negatives" (correct history, wrong immediate action)
                if len(low_act_list) > 0:
                    last_act = low_act_list[-1]
                    high_plan_seq_rejected = ""
                    
                    # Copy all but last
                    for _, low_act in enumerate(low_act_list[:-1]):
                        action_str = self.serialize_action(low_act['api_action'])
                        high_plan_seq_rejected += (self.img_token + action_str)
                    
                    # Generate hallucinated last action
                    true_act_name = last_act['api_action']['action']
                    possible_negatives = [a for a in self.all_actions if a != true_act_name]
                    bad_act_name = random.choice(possible_negatives) if possible_negatives else "NoOp"
                    
                    fake_api = {'action': bad_act_name}
                    
                    # 2. Fill slots with REAL objects from the scene
                    # If scene_objects is empty (rare), fall back to a dummy
                    fallback_obj = "random_object|001"
                    
                    if '[object]' in self.act_template[bad_act_name]:
                        fake_api['objectId'] = random.choice(scene_objects) if scene_objects else fallback_obj
                    
                    if '[receptacle]' in self.act_template[bad_act_name]:
                        fake_api['receptacleObjectId'] = random.choice(scene_objects) if scene_objects else fallback_obj

                    bad_action_str = self.serialize_action(fake_api)
                    high_plan_seq_rejected += (self.img_token + bad_action_str)
                else:
                    high_plan_seq_rejected = high_plan_seq_chosen # Fallback

                # --- Package Chunk ---
                # Context so far + Current Chosen Step
                full_chosen = chunk_seq_base + high_plan_seq_chosen
                # Context so far + Current Rejected Step
                full_rejected = chunk_seq_base + high_plan_seq_rejected
                
                # Images: Context Images + Current Step Images
                full_imgs = chunk_img_files + step_imgs
                
                # Yield this step as a training sample
                # NOTE: In strict DPO, you might want shorter contexts to save memory, 
                # or full context. This appends full history.  

                # --- Update Rolling Context (Always assume Expert path was taken) ---
                chunk_seq_base += high_plan_seq_chosen
                chunk_img_files.extend(step_imgs)
                
                # # Check buffer size to prune history if needed (simplified from original)
                # if len(chunk_img_files) > 40: # simple sliding window example
                #      # Reset context if too long for simple logic
                #      chunk_img_files = chunk_img_files[-1:] 
                #      chunk_seq_base = main_goal_str + self.img_token # Rough reset
            chunk_output.append((full_chosen, full_rejected, full_imgs))

        return chunk_output
    
    def serialize_action(self, act):
        template = self.act_template[act['action']]
        if 'objectId' in act:
            template = template.replace("[object]", act['objectId'].split("|")[0])
        if 'receptacleObjectId' in act:
            template = template.replace("[receptacle]", act['receptacleObjectId'].split("|")[0])
        return '<|act|>' + template + '<|act|>'


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
        # Batch is list of dicts: {chosen_input_ids, rejected_input_ids, pixel_values...}
        max_img_len = max(sample['pixel_values'].size(0) for sample in batch)
        
        c_input_ids = []
        c_labels = []
        r_input_ids = []
        r_labels = []
        pixel_values = []
        image_grid_thw = []
        
        for sample in batch:
            c_input_ids.append(sample['chosen_input_ids'])
            c_labels.append(sample['chosen_labels'])
            r_input_ids.append(sample['rejected_input_ids'])
            r_labels.append(sample['rejected_labels'])

            # Image padding
            pad_len = max_img_len - sample['pixel_values'].size(0)
            if pad_len > 0:
                pad_shape = (pad_len, *sample['pixel_values'].shape[1:])
                padding = torch.zeros(pad_shape, dtype=sample['pixel_values'].dtype)
                pixel_values.append(torch.cat([sample['pixel_values'], padding], dim=0))
            else:
                pixel_values.append(sample['pixel_values'])
            
            image_grid_thw.append(sample['image_grid_thw'])

        batch_dict = {
            'chosen_input_ids': torch.concat(c_input_ids, dim=0),
            'chosen_labels': torch.concat(c_labels, dim=0),
            'rejected_input_ids': torch.concat(r_input_ids, dim=0),
            'rejected_labels': torch.concat(r_labels, dim=0),
            'pixel_values': torch.concat(pixel_values, dim=0),
            'image_grid_thw': torch.concat(image_grid_thw, dim=0),
        }

        return batch_dict