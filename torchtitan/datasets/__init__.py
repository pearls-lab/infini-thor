# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.datasets.hf_datasets import build_hf_data_loader
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.datasets.processor import build_hf_processor
from torchtitan.logging import logger


def build_data_loader(
    job_config,
    processor,
    split="train",
    rank=0,
    dp_world_size=1,
    dp_rank=0,
    img_token_id=None,
):
    """Build a data loader based on the dataset specified in job_config.training.dataset."""
    dataset_name = job_config.training.dataset

    if dataset_name == "alfred_dataset":
        from torchtitan.datasets.alfred_dataset_long_ctx import ALFREDDataset, AlfredDataLoader

        dataset = ALFREDDataset(
            processor=processor,
            traj_data_dir=job_config.training.traj_data_dir,
            img_data_dir=job_config.training.img_data_dir,
            split=split,
            max_seq_len=job_config.training.seq_len,
            world_size=dp_world_size,
            cp_degree=job_config.experimental.context_parallel_degree,
            rank=rank,
        )
        data_loader = AlfredDataLoader(
            dp_rank=dp_rank,
            hf_ds=dataset,
            batch_size=job_config.training.batch_size,
            world_size=dp_world_size,
        )
        return data_loader
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: ['alfred_dataset']"
        )


__all__ = [
    "build_hf_data_loader",
    "build_tokenizer",
    "build_hf_processor",
    "build_data_loader",
]
