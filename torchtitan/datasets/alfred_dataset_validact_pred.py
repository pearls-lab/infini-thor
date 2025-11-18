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


def read_images(img_dir, img_width, img_height):
    image_dict = {}
    for filename in os.listdir(img_dir):
        img_file = os.path.join(img_dir, filename)
        image = Image.open(img_file)  # PIL can open file paths directly
        image = image.convert("RGB")
        if image.size != (img_width, img_height):
            image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        image_dict[filename] = image

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
        self.ignore_index = ignore_index
        self.eval = eval
        self.pad_to = pad_to
        self.rank = rank
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        
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

        self.system_prompt = (
            "You are an embodied AI agent operating in the simulated 3D household environment AI2-THOR. "
            "Your goal is to predict the NEXT VALID ACTIONS and the BEST ACTION, given the task goal, the sequence of prior actions, "
            "and the current ego-centric visual observation. Always use the latest image (current state) when making decisions.\n\n"

            "Available actions: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, "
            "OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject.\n\n"

            "=== ACTION CONSTRAINTS (must match the CURRENT image/state) ===\n"
            "- Navigation (MoveAhead / RotateLeft / RotateRight): MoveAhead only if the path is clear. If blocked or facing a wall, rotate first.\n"
            "- PickupObject: Only if the target object is visible and reachable.\n"
            "- PutObject: Only if you are holding an object AND a suitable receptacle/place is directly in front of you.\n"
            "- OpenObject / CloseObject: Only for openable objects (Cabinet, Fridge, Drawer, etc.) visible and within reach.\n"
            "- SliceObject: Only if you are currently holding a Knife or ButterKnife.\n"
            "- ToggleObjectOn / ToggleObjectOff: Only for togglable objects (e.g., Faucet, Microwave, Lamp) visible and within reach.\n"
            "- Do NOT propose impossible actions (e.g., moving through obstacles) or duplicate actions.\n\n"

            "=== OUTPUT FORMAT ===\n"
            "- First, output all valid actions as a comma-separated list with no explanations.\n"
            "- Then, output the best action among them.\n"
            "- For object interactions, follow EXACTLY these formats (case-sensitive):\n"
            "    OpenObject [object]\n"
            "    CloseObject [object]\n"
            "    PickupObject [object]\n"
            "    PutObject [object] [receptacle]\n"
            "    ToggleObjectOn [object]\n"
            "    ToggleObjectOff [object]\n"
            "    SliceObject [object]\n\n"

            "=== EXAMPLES ===\n"
            "    next valid actions: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, best action: MoveAhead\n"
            "    next valid actions: RotateLeft, RotateRight, LookUp, LookDown, PickupObject Mug, best action: PickupObject Mug\n"
            "    next valid actions: RotateLeft, RotateRight, LookDown, PutObject Apple CounterTop, best action: PutObject Apple CounterTop\n"
            "    next valid actions: RotateLeft, RotateRight, LookUp, OpenObject Fridge, best action: OpenObject Fridge\n"
        )

        if len(self.data) == 0:
            self._load_data()

    def __len__(self):
        return self.get_num_samples()

    def get_num_samples(self):
        num_sample = 0
        for traj in self.traj_data:
            num_sample += len(traj['validact_pair'])
        return num_sample

    def _get_data_iter(self):
        it = iter(self.data)
        for _ in range(self._traj_idx): # iterator starting at sample_idx (if sample_idx is not 0 from the dataloader state)
            next(it)
        return it

    def _traj_seed(self, filename: str) -> int:
        """Deterministic per-trajectory seed shared across ranks."""
        h = hashlib.md5(filename.encode("utf-8")).hexdigest()
        # Take first 8 hex digits -> 32-bit int
        return int(h[:8], 16)

    def filter_samples(self, validact_pair, generated_actions, filename: str):
        nav_acts = set(["RotateRight", "RotateLeft", "LookUp", "LookDown"])
        filtered_samples = []
        nav_samples = []
        for si, (seq, valact) in enumerate(validact_pair.items()):
            act_seq = [{'action': "INIT"}] + generated_actions[:si]
            best_act = generated_actions[si]
            act_set = set([x['action'] for x in valact])
            if (act_set & nav_acts) == nav_acts:
                nav_samples.append((act_seq, valact, best_act))
            else:
                filtered_samples.append((act_seq, valact, best_act))
        
        # random.shuffle(nav_samples)
        # filtered_samples += nav_samples[:int(len(nav_samples)*0.2)]
        # print(f"\t\t filter out: {len(validact_pair)} -> {len(filtered_samples)}")
        # return filtered_samples

        # --- deterministic shuffle: same on all DP ranks ---
        keep_n = int(len(nav_samples) * 0.2)

        if keep_n > 0:
            rng = random.Random(self._traj_seed(filename))
            indices = list(range(len(nav_samples)))
            rng.shuffle(indices)
            for idx in indices[:keep_n]:
                filtered_samples.append(nav_samples[idx])

        print(f"\t\t filter out: {len(validact_pair)} -> {len(filtered_samples)}")
        return filtered_samples

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
            # Always advance sample cursor so we can't get stuck if we skip
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

            # TODO: filter out - use only 20% for major samples
            filtered_sample = self.filter_samples(traj['validact_pair'], traj['generated_actions'], filename)
            N = len(filtered_sample)
            usable = (N // dp_world) * dp_world
            
            for si, (seq, valact, best_act) in enumerate(filtered_sample):
                if si >= usable:
                    break
                
                self._sample_idx += 1

                # Keep only trajectories owned by this shard
                if (si % dp_world) != dp_rank:
                    continue

                # Heavy work happens ONLY for this shard's trajectories
                #best_act = traj['generated_actions'][si]
                #act_seq = seq.split("||")
                
                #content, assistant_response, img_list = self._load_sample([{'action': "INIT"}] + traj['generated_actions'][:si], valact, best_act, img_dict, lowidx2img)
                content, assistant_response, img_list = self._load_sample(seq, valact, best_act, img_dict, lowidx2img)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": assistant_response}
                ]

                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,   # set True if you plan to .generate immediately
                )

                output = self.processor(text=prompt, images=img_list, return_tensors="pt")

                #print(f"[rank{self.rank}][dp_rank{self.dp_rank}] traj_idx: {self._traj_idx} sample_idx: {self._sample_idx} n_img: {len(img_list)}\nprompt: {prompt}")
                
                labels = output.input_ids.clone()
                
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
                    # Fallback: try finding a partial match or key phrase
                    logger.warning(f"Could not find exact assistant token match")
                    # You might want to handle this case differently
                    assistant_start_idx = 0
                

                # Default: mask all; if anchor found, unmask only the target span (after anchor)
                labels[:] = self.ignore_index
                labels[0, assistant_start_idx:] = output.input_ids[0, assistant_start_idx:]

                shift_input_ids = output.input_ids[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # input_ids = pad_to_multiple(input_ids, self.pad_to, pad_token=self.eos_tok_id)
                # labels = pad_to_multiple(labels, self.pad_to, pad_token=self.ignore_index)
                logger.info(f"[rank{self.rank}][dp_rank{self.dp_rank}] traj_idx: {self._traj_idx} sample_idx: {self._sample_idx} input_ids: {output.input_ids.shape} n_img: {len(img_list)}")
                logger.info(f"[rank{self.rank}][dp_rank{self.dp_rank}] labels: {self.processor.tokenizer.decode(labels[0, assistant_start_idx:]).strip()}")
                # print(f"[rank{self.rank}][dp_rank{self.dp_rank}] traj_idx: {self._traj_idx} sample_idx: {self._sample_idx} input_ids: {output.input_ids.shape} n_img: {len(img_list)}")
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

    def get_act_str(self, raw_act_text):
        if "PutObject" in raw_act_text:
            act_toks = raw_act_text.split("_")
            obj_id = act_toks[1].split("|")[0]
            recep_id = act_toks[2].split("|")[0]
            return f"{act_toks[0]} {obj_id} {recep_id}"
        elif "Object" in raw_act_text:
            act_toks = raw_act_text.split("_")
            obj_id = act_toks[1].split("|")[0]
            return f"{act_toks[0]} {obj_id}"
        else:
            return raw_act_text

    def get_act_str2(self, act_dict):
        if act_dict['action'] == 'PutObject':
            return f"PutObject {act_dict['objectId']} {act_dict['receptacleObjectId']}"
        elif 'Object' in act_dict['action']:
            return f"{act_dict['action']} {act_dict['objectId']}"
        else:
            return act_dict['action']

    def get_obj_name(self, obj_id):
        return obj_id.split("|")[0]

    def act_dict_to_str(self, va):
        if va['action'] == 'PutObject':
            return f"PutObject {self.get_obj_name(va['objectId'])} {self.get_obj_name(va['receptacleObjectId'])}"
        elif 'Object' in va['action']:
            return f"{va['action']} {self.get_obj_name(va['objectId'])}"
        else:
            return va['action']

    def val_act_list_to_str(self, valact):
        _act_list = [self.act_dict_to_str(va) for va in valact]
        _act_set = set(_act_list)
        return ", ".join(_act_set)

    def _load_sample(self, seq_list, valact, best_act, img_dict, lowidx2img):
        contents = []
        imgs = []
        
        for low_idx, act in enumerate(seq_list):
            if act['action'] == "INIT":
                contents.append({"type": "text", "text": f"initial state: "})
            else:
                contents.append({"type": "text", "text": f" action: {self.get_act_str2(act)} state: "})
            #contents.append({"type": "image", "image": f'{img_idx:06d}.png'})
            for low_img_name in lowidx2img[low_idx]:
                contents.append({"type": "image", "image": img_dict[low_img_name]})
                imgs.append(img_dict[low_img_name])
        
        MAX_IMGS = 280

        if len(imgs) > MAX_IMGS:
            # keep last MAX_IMGS images
            imgs = imgs[-MAX_IMGS:]

            # Now we must update contents accordingly:
            # keep all text, but prune image entries from the front.
            new_contents = []
            remaining = len(imgs)
            
            for c in reversed(contents):
                if c["type"] == "image":
                    if remaining > 0:
                        new_contents.append(c)
                        remaining -= 1
                    # else drop
                else:
                    # always keep text
                    new_contents.append(c)

            # reverse again to restore original order
            contents = list(reversed(new_contents))

        assistant_response = (
            f"next valid actions: {self.val_act_list_to_str(valact)}, "
            f"best action: {self.act_dict_to_str(best_act)}"
        )
        
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


class SortishPool(IterableDataset):
    def __init__(self, base_ds, pool_size=512):
        super().__init__()
        self.base_ds = base_ds
        self.pool_size = pool_size

    def __iter__(self):
        pool = []
        for sample in self.base_ds:
            L = sample['input_ids'].shape[1]  # length before any padding
            pool.append((L, sample))
            if len(pool) >= self.pool_size:
                pool.sort(key=lambda x: x[0])  # ascending lengths
                for _, s in pool:
                    yield s
                pool.clear()
        # flush tail
        if pool:
            pool.sort(key=lambda x: x[0])
            for _, s in pool:
                yield s


class BucketByLen(IterableDataset):
    def __init__(self, base_ds, length_key=None):
        super().__init__()
        self.base_ds = base_ds
        self.bounds = [128,256,512,1024,2048,4096,8192,16384,32768]
        self.per_bucket_bs = {
            128: 32,
            256: 24,
            512: 16,
            1024: 12,
            2048: 8,
            4096: 4,
            8192: 2,
            16384: 2,
            32768: 1
        }
        self.buckets = {b: [] for b in self.bounds}
        self.drop_last = False
        self.length_key = length_key or (lambda s: s['input_ids'].shape[1])

    def __len__(self):
        return self.base_ds.get_num_samples()

    def _pick_bound(self, L: int):
        for b in self.bounds:
            if L <= b:
                return b
        return self.bounds[-1]

    def __iter__(self):
        buffers = {b: [] for b in self.bounds}
        for s in self.base_ds:
            L = self.length_key(s)
            b = self._pick_bound(L)
            buffers[b].append(s)
            if len(buffers[b]) >= self.per_bucket_bs[b]:
                # yield a *variable-size batch* (list of samples)
                yield buffers[b]
                buffers[b] = []

        # flush the tail
        if not self.drop_last:
            for b in self.bounds:
                if buffers[b]:
                    yield buffers[b]


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

    @staticmethod
    def collate_fn_backup(batch, eos_id, ignore_index=-100):
        # figure out per-batch max sequence length
        # inspection:
        # len(batch) = 1
        # len(batch[0]) = 4
        # type(batch[0][0]): dict, len(batch[0][0]) = 4
        # batch[0][0]['input_ids'].shape: torch.Size([1, 2196])
        # batch[0][1]['input_ids'].shape: torch.Size([1, 2349])
        # batch[0][2]['input_ids'].shape: torch.Size([1, 2499])
        # batch[0][3]['input_ids'].shape: torch.Size([1, 2655])
        batch = batch[0] # the actual batch size is handled in Dataset level
        max_len = max(s['input_ids'].shape[1] for s in batch)

        # TODO
        # if you need alignment to multiples:
        # pad_to = getattr(batch[0], 'pad_to', 1)  # or pass in via dataset
        # if pad_to > 1:
        #     extra = (pad_to - (max_len % pad_to)) % pad_to
        #     max_len += extra

        # image count padding (you already do this)
        max_img_len = max(sample['pixel_values'].size(0) for sample in batch)

        input_ids, labels, pixel_values, n_image, image_grid_thw = [], [], [], [], []

        # grab pad tokens from first sample’s metadata
        # eos_id = batch[0].get('eos_tok_id', None)
        # ignore_id = batch[0].get('ignore_index', -100)

        for sample in batch:
            # pad text
            input_ids.append(AlfredValidActDataLoader._pad_1d(
                sample['input_ids'], max_len, eos_id
            ))
            if 'labels' in sample:
                labels.append(AlfredValidActDataLoader._pad_1d(
                    sample['labels'], max_len, ignore_index
                ))

            # pad images along frame dimension
            pad_len = max_img_len - sample['pixel_values'].size(0)
            if pad_len > 0:
                pad_shape = (pad_len, *sample['pixel_values'].shape[1:])
                padding = torch.zeros(pad_shape, dtype=sample['pixel_values'].dtype)
                pixel_values.append(torch.cat([sample['pixel_values'], padding], dim=0))
            else:
                pixel_values.append(sample['pixel_values'])

            n_image.append([sample['pixel_values'].shape[0]])
            image_grid_thw.append(sample['image_grid_thw'])

        batch_dict = {
            'input_ids': torch.cat(input_ids, dim=0),
            'pixel_values': torch.cat(pixel_values, dim=0),
            'n_image': torch.tensor(n_image, dtype=input_ids[0].dtype),
            'image_grid_thw': torch.cat(image_grid_thw, dim=0),
        }
        if labels:
            batch_dict['labels'] = torch.cat(labels, dim=0)

        # # (optionally) stash pad tokens for downstream
        # batch_dict['eos_tok_id'] = eos_id
        # batch_dict['ignore_index'] = ignore_id
        return batch_dict
