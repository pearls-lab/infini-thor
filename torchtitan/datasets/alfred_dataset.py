import os
import re
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.logging import logger
#from torchtitan.datasets.hf_datasets import DPAwareDataLoader

from datasets import Dataset, load_dataset

from PIL import Image
import tarfile
from io import BytesIO


# def extract_and_convert_tar(tar_path):
#     """Extracts a .tar file and converts all .jpg files inside to a list of PIL images."""
#     pil_images = []
    
#     with tarfile.open(tar_path, 'r') as tar:
#         for member in tar.getmembers():
#             if member.isfile() and member.name.lower().endswith(".jpg"):
#                 file_obj = tar.extractfile(member)
#                 if file_obj:
#                     image = Image.open(BytesIO(file_obj.read()))
#                     image = image.convert("RGB")  # Ensure consistent format
#                     pil_images.append(image)
    
#     return pil_images

def extract_and_convert_tar(tar_path):
    """Extracts a .tar file and converts all .jpg files inside to a dictionary where keys are filenames and values are PIL images."""
    image_dict = {}
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(".jpg"):
                file_obj = tar.extractfile(member)
                if file_obj:
                    base_filename = os.path.basename(member.name)
                    image = Image.open(BytesIO(file_obj.read()))
                    image = image.convert("RGB")  # Ensure consistent format
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


class ALFREDDataset(IterableDataset, Stateful):

    def __init__(
        self,
        processor,
        traj_data_dir: str = "",
        img_data_dir: str = "",
        split: str = "train",
        max_seq_len: int = 131072,
        world_size: int = 1,
        cp_degree: int = 1,
        rank: int = 0,
        infinite: bool = False,
        ignore_index: int = -100,
        eval: bool = False
    ) -> None:
        self.dataset_name = "alfred"
       
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.infinite = infinite
        self.img_tok_id = processor.tokenizer('<image>').input_ids[0]
        self.act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
        self.eos_tok_id = processor.tokenizer.eos_token_id
        self.ignore_index = ignore_index
        self.eval = eval
        self.world_size = world_size
        self.cp_degree = cp_degree

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

        if len(self.traj_data) == 0:
            self._load_traj_data()

    def __len__(self):
        return len(self.traj_data)

    def _get_data_iter(self):
        if self._sample_idx == len(self.traj_data):
            self._sample_idx = 0
            self._chunk_idx = 0

        it = iter(self.traj_data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        for traj in self._get_data_iter():
            print(f"Loading a new example ... ")

            if self.eval:
                yield json.loads(traj['text']) 
            else:
                chunks = self._load_sample(traj, chunk=True)

                if not isinstance(chunks, list):
                    chunks = [chunks]

                for ci, chunk in enumerate(chunks[self._chunk_idx:]):
                    n_img_token = chunk['lang_input'].count('<image>')
                    n_act_token = chunk['lang_input'].count('<|act|>')
                    if n_act_token == 0:
                        logger.warning(f"Skip this chunk - no target labels (action tokens)")
                        continue
                    if not self.use_only_last_frame and len(chunk['img_list']) != n_img_token:
                        logger.warning(f"Some images are missed -- expected {n_img_token}, but {len(chunk['img_list'])}")
                        logger.warning(f"len(chunk['img_list']): {len(chunk['img_list'])}, len(chunk['lang_input']): {len(chunk['lang_input'])}, chunk['lang_input']: {chunk['lang_input']}")
                        continue # raise ValueError()

                    output = self.processor(images=chunk['img_list'], text=chunk['lang_input'], return_tensors="pt")

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
                    
                    if self.cp_degree > 1:
                        input_ids = pad_to_multiple(input_ids, self.cp_degree * 2, pad_token=self.eos_tok_id)
                        labels = pad_to_multiple(labels, self.cp_degree * 2, pad_token=self.ignore_index)
                    else:
                        input_ids = pad_to_max_seq(input_ids, max_seq=self.max_seq_len, pad_token=self.eos_tok_id)[:, :self.max_seq_len]
                        labels = pad_to_max_seq(labels, max_seq=self.max_seq_len, pad_token=self.ignore_index)[:, :self.max_seq_len]
                        
                    self._chunk_idx += 1
                    yield {
                        'input_ids': input_ids,
                        'pixel_values': output.pixel_values, 
                        'image_sizes': output.image_sizes,
                        'labels': labels
                    }
                    
                # reset chunk_idx
                self._chunk_idx = 0
            self._sample_idx += 1

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
        chunk_seq_list, chunk_img_list = self.seq_preprocess(traj)

        #traj_imgs = set([x['image_name'].split(".")[0] for x in traj['images']])

        img_tar_file = filename.replace("txt", "tar")
        tar_file = os.path.join(self.img_data_dir, img_tar_file)

        img_dict = extract_and_convert_tar(tar_file)

        chunks = []

        if self.use_only_last_frame:
            for input_seq, cimgs in zip(chunk_seq_list, chunk_img_list):
                chunks.append({
                    'lang_input': input_seq,
                    'img_list': [img_dict[fname.replace("png", "jpg")] for fname in cimgs],
                    'task_goal': traj['turk_annotations']['anns'][0]['task_desc'],
                    'traj': traj,
                })
        else:
            for input_seq, (img_start, img_end) in zip(chunk_seq_list, chunk_img_idx):
                chunks.append({
                    'lang_input': input_seq,
                    'img_list': img_list[img_start:img_end],
                    'task_goal': traj['turk_annotations']['anns'][0]['task_desc'],
                    'traj': traj,
                })

        return chunks

    def _load_traj_data(self):
        directory_path = self.traj_data_dir
        all_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    all_files.append((file_path, file))
        
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
        main_goal_str = "<|goal|>Your main goal: "
        if 'turk_annotations' in traj:
            main_goal_str += traj['turk_annotations']['anns'][0]['task_desc'] + "<|goal|>"
        # else we need to use templated desc .. later
        n_main_goal_tokens = len(self.processor(text=main_goal_str).input_ids)

        chunk_seq_list = []
        chunk_img_file_list = []

        chunk_seq = main_goal_str
        n_chunk_tokens = n_main_goal_tokens # for chunking

        # initial image state
        chunk_seq += "<image>"
        n_chunk_tokens += 1485
        chunk_img_files = ['000000000.png']
        n_chunk_img = 1
        img_start_idx = 1

        for high_idx, low_act_list in high_idx_2_low_act_list.items():
            plan_str = f"<|plan|>Plan: {self.get_templated_high_pddl_desc(traj['plan']['high_pddl'][high_idx])}<|plan|>"
            
            high_plan_seq = ""
            high_plan_seq += plan_str
            n_high_plan_tokens = len(self.processor(text=plan_str).input_ids)
            n_high_plan_img = 0

            low_act_last_frames = []
            #for low_idx, low_act in enumerate(low_act_list):
            for _, low_act in enumerate(low_act_list):
                low_idx = low_act['low_idx']
                low_act_last_frames.append(low_idx_2_image[low_idx][-1])
                action_str = self.serialize_action(low_act['api_action'])
                low_act_seq = action_str
                action_str_tok = self.processor(text=action_str).input_ids   
                n_low_act_tokens = len(action_str_tok)

                # count tokens for images
                n_low_img = len(low_idx_2_image[low_idx])

                if self.use_only_last_frame:
                    low_act_seq += ("<image>" * 1)
                    n_low_act_tokens += (1485 * 1) # one frame is 1485 tokens
                else:
                    low_act_seq += ("<image>" * n_low_img)
                    n_low_act_tokens += (1485 * n_low_img) # one frame is 1485 tokens

                if (n_high_plan_tokens + n_low_act_tokens) >= self.max_seq_len:
                    break # do not add this low_act and break
                else:
                    n_high_plan_tokens += n_low_act_tokens
                    high_plan_seq += low_act_seq
                    n_high_plan_img += n_low_img

            assert n_high_plan_tokens < self.max_seq_len

            if (n_chunk_tokens + n_high_plan_tokens) >= self.max_seq_len:
                chunk_seq_list.append(chunk_seq)
                chunk_img_file_list.append(chunk_img_files)
                
                # reset for next chunk
                chunk_seq = main_goal_str + high_plan_seq
                n_chunk_tokens = n_main_goal_tokens + n_high_plan_tokens
                img_start_idx = img_start_idx + n_chunk_img
                n_chunk_img = n_high_plan_img
                chunk_img_files = low_act_last_frames
            else:
                chunk_seq += high_plan_seq
                n_chunk_tokens += n_high_plan_tokens
                n_chunk_img += n_high_plan_img
                chunk_img_files.extend(low_act_last_frames)

        chunk_seq_list.append(chunk_seq)
        chunk_img_file_list.append(chunk_img_files)

        if self.use_only_last_frame:
            assert len(chunk_seq_list) == len(chunk_img_file_list)
        #logger.info(f"# of chunks: {len(chunk_seq_list)}, chunk_img_file_list: {chunk_img_file_list[-1]}")

        return chunk_seq_list, chunk_img_file_list

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


class AlfredDataLoader(StatefulDataLoader, Stateful):
    # TODO: Update Dataset class with DP-aware shard and remove the wrapper (ShardedDataset)
    
    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int, world_size: int):
        # super().__init__(hf_ds, batch_size, collate_fn=self.collate_fn)
        # Wrap the dataset with a DP-aware shard
        dp_sharded_dataset = self.shard_dataset(hf_ds, dp_rank, world_size)
        super().__init__(dp_sharded_dataset, batch_size, collate_fn=self.collate_fn, drop_last=True)

    def shard_dataset(self, dataset: IterableDataset, dp_rank: int, world_size: int):
        # Shard the dataset across DP ranks
        class ShardedDataset(IterableDataset):
            def __iter__(self_inner):
                all_data_iter = iter(dataset)
                # Offset the iterator so each rank sees disjoint samples
                return (x for i, x in enumerate(all_data_iter) if i % world_size == dp_rank)

        return ShardedDataset()

    def load_state_dict(self, state_dict):
        if isinstance(self.dataset, Stateful):
            # Resuming dataloader with: {'_index_sampler_state': None, '_sampler_iter_state': {'samples_yielded': 1480}, '_sampler_iter_yielded': 740, '_num_yielded': 740, '_IterableDataset_len_called': None, '_shared_seed': None, 'fetcher_state': {'dataset_iter_state': None, 'fetcher_ended': False}, 'dataset_state': None, '_iterator_finished': False}
            logger.info(f"Resuming dataloader with: {state_dict}")
            self.dataset.load_state_dict(state_dict)

    @staticmethod
    def collate_fn(batch):
        max_img_len = max(sample['pixel_values'].size(0) for sample in batch)
        
        input_ids = []
        pixel_values = []
        n_image = []
        labels = []
        
        for sample in batch:
            input_ids.append(sample['input_ids'])

            pad_len = max_img_len - sample['pixel_values'].size(0)

            if pad_len > 0:
                pad_shape = (pad_len, *sample['pixel_values'].shape[1:])
                padding = torch.zeros(pad_shape, dtype=sample['pixel_values'].dtype, 
                                device=sample['pixel_values'].device)
                pixel_values.append(torch.cat([sample['pixel_values'], padding], dim=0))
            else:
                pixel_values.append(sample['pixel_values'])

            n_image.append([sample['pixel_values'].shape[0]])
            labels.append(sample['labels'])

        batch_dict = {
            'input_ids': torch.concat(input_ids, dim=0),
            'pixel_values': torch.stack(pixel_values),
            'n_image': torch.tensor(n_image, device=input_ids[0].device, dtype=input_ids[0].dtype)
        }
        
        if labels:
            batch_dict['labels'] = torch.concat(labels, dim=0)
            
        return batch_dict
