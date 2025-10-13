# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    get_schedule_class,
    ScheduleZBVZeroBubble,
)
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from torchtitan.models.llama.model import TransformerModelArgs
from torchtitan.parallelisms import ParallelDims
from torchtitan.parallelisms.pipeline import (
    build_pipeline_schedule,
    generate_split_points,
    stage_ids_this_rank,
)

DeviceType = Union[int, str, torch.device]


def pipeline_llava(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: TransformerModelArgs,
    loss_fn: Callable[..., torch.Tensor],
):
    stages, models = pipeline_llava_manual_split(
        model, pp_mesh, parallel_dims, job_config, device, model_config
    )

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, models, has_first_stage, has_last_stage


def pipeline_llava_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: TransformerModelArgs,
):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    splits = (
        job_config.experimental.pipeline_parallel_split_points
        or generate_split_points(job_config, parallel_dims.pp, model_config.text_config.num_hidden_layers)
    )
    logger.info(f"splits: {splits}")

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        logger.info(f"stage_idx: {stage_idx}, start_layer: {start_layer}, stop_layer: {stop_layer}, is_first: {is_first}, is_last: {is_last}, len(layers): {len(model.language_model.model.layers)}")

        if not is_first:
            model.language_model.model.embed_tokens = None
            del model.vision_tower
            del model.multi_modal_projector
        
        drop_layers = start_layer is not None

        layers = model.language_model.model.layers

        if isinstance(layers, nn.ModuleList):
            start_idx = int(start_layer.split(".")[-1]) if start_layer else 0
            stop_idx = int(stop_layer.split(".")[-1]) if stop_layer else len(layers)
            model.language_model.model.layers = nn.ModuleList(
                layers[start_idx:stop_idx]
            )
        elif isinstance(layers, nn.ModuleDict):
            drop_layers = start_layer is not None
            layers_to_delete = []

            for name in layers.keys():
                layer_name = f"layers.{name}"
                if layer_name == start_layer:
                    drop_layers = False
                if drop_layers:
                    layers_to_delete.append(name)
                if layer_name == stop_layer:
                    drop_layers = True

            for name in layers_to_delete:
                del layers[name]
                
        '''
        if isinstance(layers, nn.ModuleDict):
            layer_iterator = layers.items() # ModuleDict case - use .items()
        else:
            layer_iterator = enumerate(layers) # ModuleList case - use enumerate

        layers_to_delete = []

        # for name in list(model.language_model.model.layers.keys()): # name is index 0, 1, 2, ...
        for name, _ in layer_iterator: # name is index 0, 1, 2, ...
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            # if drop_layers:
            #     del model.language_model.model.layers[name]
            if drop_layers:
                layers_to_delete.append(name)

        for name in layers_to_delete:
            del model.language_model.model.layers[name]
        '''
        if not is_last:
            model.language_model.model.norm = None
            model.language_model.lm_head = None

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []

    schedule_class = get_schedule_class(
        job_config.experimental.pipeline_parallel_schedule
    )
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style=style):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models
