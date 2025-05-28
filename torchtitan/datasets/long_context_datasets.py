import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from scripts.long_context.eval.needle.utils import load_context, insert_needle


class MyDataset(IterableDataset, Stateful):
    def __init__(
        self,
        #data: List,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        self.dataset_name = "long_conext_dataset"
       
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

        needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        depth = 0.5
        _context = load_context(fpath="scripts/long_context/eval/needle/PaulGrahamEssays/*.txt", ctx_len=seq_len)
        context = ""
        # # of entire context tokens = 148836
        for _ in range((seq_len//148836)+1):
            context += _context

        context = insert_needle(context, needle, depth=depth)
        needle_idx = context.find("The best thing to do in San Francisco is")
        logger.info("Context has %d chars, needle inserted at %d char location:\n" % (len(context), needle_idx))
        logger.info(context[needle_idx - 150: needle_idx + 150]) # look at how the needle is inserted 
        prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context
        question = "What is the best thing to do in San Francisco?"
        prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
        self._data = [prompt]

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        while True:
            for sample_text in self._get_data_iter():
                if isinstance(self._tokenizer, PreTrainedTokenizerFast):
                    sample_tokens = self._tokenizer.encode(sample_text)
                else:
                    sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                
                self._sample_idx += 1

                x = torch.LongTensor(sample_tokens)
                    
                yield x

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}
