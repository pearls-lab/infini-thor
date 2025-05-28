import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import init_logger, logger

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

import requests
from PIL import Image

init_logger()


class MyDataset(IterableDataset, Stateful):
    def __init__(
        self,
        processor,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        self.dataset_name = "toy_dataset"  
        self.processor = processor
        self._data = ["test"]

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        for d in self._data:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What animal is shown in this image? and how many the animals are?"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
            raw_image = Image.open(requests.get(image_file, stream=True).raw)
            inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            yield inputs
