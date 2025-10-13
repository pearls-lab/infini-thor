#!/usr/bin/env python3
# Copyright (c) Meta Platforms
# Unified CP/TP/PP training script for LLaVA- and Qwen2-VL-style models using torchtitan.
# This merges the core structure of the two provided scripts and keeps context-parallelism.

import os
import time
from datetime import timedelta
from pathlib import Path
import subprocess
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import distribute_module, distribute_tensor, DTensor, Replicate, Shard

# --- torchtitan imports (as used in your scripts) ---
from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader, build_hf_processor
from torchtitan.tools.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_tokenizer
from torchtitan.parallelisms import ParallelDims
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.train_spec import get_train_spec
from torchtitan.utils import device_module, device_type, import_module_from_path

from huggingface_hub import snapshot_download, upload_folder, create_repo

AWS_S3_PATH = os.environ.get('AWS_S3_PATH', None)


# -----------------------------
# Helpers preserved from originals
# -----------------------------

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def set_nested_attr(obj, name, value):
    """Register a buffer on nested modules (works through lists/ModuleLists), like your originals."""
    parts = name.split('.')
    if not hasattr(obj, parts[0]):
        # In PP, some parts (e.g., vision_tower) may not exist on all stages.
        return
    for part in parts[:-1]:
        if isinstance(obj, (nn.ModuleList, list)):
            idx = int(part)
            if idx < len(obj):
                obj = obj[idx]
            else:
                logger.info(f"register buffer: PP applied — this model part lacks layers.{idx}")
                return
        else:
            obj = getattr(obj, part)
    obj.register_buffer(parts[-1], value)

def combine_model_parts_state(model_parts: List[nn.Module]):
    out = {}
    for m in model_parts:
        sd = m.state_dict()
        for k, v in sd.items():
            if v is not None:
                out[k] = v
    return out

def save_checkpoint_s3(states, step, output_dir):
    """Kick off an async S3 sync from rank 0 and barrier the group (kept from originals)."""
    if AWS_S3_PATH and get_local_rank() == 0:
        try:
            sync_command = f"nohup aws s3 sync {output_dir} {AWS_S3_PATH}/step-{step} > /tmp/s3_sync_{step}.log 2>&1 &"
            subprocess.Popen(sync_command, shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            logger.info(f"Started background S3 sync for checkpoint at step {step}")
        except Exception as e:
            logger.error(f"Error starting S3 sync: {e}")
    dist.barrier()

def warmup_dynamic_rope_scaling(model, device, seq_len, rope_kwargs):
    """Matches your warm-up path for RoPE scaling to avoid on-the-fly reallocs."""
    try:
        layers = model.language_model.model.layers
        config = model.language_model.config if hasattr(model.language_model, "config") else model.config

        if rope_kwargs.get('rope_type') == "yarn":
            config.rope_scaling = rope_kwargs
            for layer in layers:
                layer.self_attn.rotary_emb.freq_update(seq_len, rope_kwargs, device=device, config=config)
            model.language_model.model.rotary_emb.freq_update(seq_len, rope_kwargs, device=device, config=config)
        else:
            for layer in layers:
                layer.self_attn.rotary_emb.freq_update(seq_len, rope_kwargs)
            model.language_model.model.rotary_emb.freq_update(seq_len, rope_kwargs)

        logger.info(f"Warmed RoPE on {len(layers)} layers (seq_len={seq_len}, rope={rope_kwargs})")
    except Exception as e:
        logger.info(f"RoPE warm-up skipped or partial: {e}")


# -----------------------------
# Training entry (keeps CP logic)
# -----------------------------

@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        import_module_from_path(job_config.experimental.custom_model_path)

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    # color printer & GC controller
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # --- distributed setup & device ---
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{get_local_rank()}")
    device_module.set_device(device)
    utils.init_distributed(job_config)

    # metrics + memory monitor
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for MFU: {gpu_peak_flops:.3e}")

    logger.info(f"ParallelDims: {parallel_dims}")

    # --- parallel meshes ---
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    pp_mesh = world_mesh["pp"] if parallel_dims.pp_enabled else None
    tp_mesh = world_mesh["tp"] if parallel_dims.tp_enabled else None

    # --- model spec & config ---
    model_name = job_config.model.name  # Expect user to set a HF ckpt (llava or qwen2-vl family)
    train_spec = get_train_spec(model_name)  # returns spec with .cls and .config mapping
    model_cls = train_spec.cls
    model_config = train_spec.config[job_config.model.flavor]

    # align model config with training args
    model_config.norm_type = job_config.model.norm_type
    #model_config.max_seq_len = job_config.training.seq_len
    text_config = getattr(model_config, "text_config", None) or model_config

    if job_config.training.rope_theta:
        text_config.rope_theta = job_config.training.rope_theta

    # sliding window attention settings for FA2
    if job_config.training.attn_impl == "flash_attention_2":
        model_config.attn_impl = "flash_attention_2"
        text_config._attn_implementation = "flash_attention_2"
        text_config.use_sliding_window = True
        text_config.max_window_layers = 0

    # --- tokenizer/processor/dataloaders ---
    processor = build_hf_processor(model_name)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})

    data_loader = build_data_loader(
        job_config,
        processor, 
        dp_mesh=dp_mesh if parallel_dims.dp_enabled else None,
        split="train",
        world_size=world_size,
        rank=dp_rank,
        img_token_id=model_config.image_token_id
    )

    # --- warm-up pass to capture buffers & set rope scaling where needed (LLaVA path) ---
    buffers_dict = None
    if "llava" in model_name.lower():
        model_tmp = model_cls.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        if job_config.training.rope_type:
            # Compute rope_kwargs similarly to your scripts
            partial_rotary_factor = getattr(text_config, "partial_rotary_factor", 1.0)
            head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            rope_kwargs = {
                "rope_type": job_config.training.rope_type,
                "factor": job_config.training.rope_factor,
                "dim": dim,
                "base": text_config.rope_theta,
                "max_position_embeddings": text_config.max_position_embeddings,
            }
            if job_config.training.rope_type == "longrope":
                rope_kwargs["long_factor"] = job_config.training.rope_factor
                rope_kwargs["short_factor"] = 1
                rope_kwargs["factor"] = 1
            if job_config.training.rope_type != "nope":
                warmup_dynamic_rope_scaling(model_tmp, device, job_config.training.seq_len, rope_kwargs)
        buffers_dict = {k: v.clone() for k, v in model_tmp.named_buffers()}
        del model_tmp
        torch.cuda.empty_cache()

    # --- meta init to control placement with TP/PP/CP ---
    with torch.device("meta"):
        if "llava" in model_name.lower() or "qwen" in model_name.lower():
            # use from_pretrained on real device later to ensure lm_head sizing vs tokenizer
            model = model_cls.from_pretrained(model_name, config=model_config, attn_implementation=job_config.training.attn_impl)
            buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        else:
            model = model_cls.from_model_args(model_config)

    # log model size
    model_param_count = utils.get_num_params(model)
    logger.info(f"Building {train_spec.name} {job_config.model.flavor} with {model_config}")
    logger.info(f"Model {model_name} size: {model_param_count:,} parameters")

    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

    # --- distribute model by PP/TP as requested ---
    model_parts = [model]

    if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
        # Distribute the module across parallel meshes as your originals do
        placements = [Replicate()]
        if parallel_dims.tp_enabled and tp_mesh is not None:
            placements = [Shard(0)]  # shard head/hidden or per your layout; placeholder kept minimal
        model = distribute_module(model, world_mesh["tp"] if parallel_dims.tp_enabled else world_mesh["dp"], placements=placements)
        model_parts = [model]
    else:
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_buffers(buffer_device=device_type, buffers_dict=buffers_dict)
        model.train()
        model_parts = [model]
        # ?
        #state_dict = {"model": m.state_dict()}
        #dcp.load(state_dict, checkpoint_id=checkpoint_path, planner=dcp.DefaultLoadPlanner(allow_partial_load=True))

    # --- optimizer/scheduler/checkpoint ---
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    start_step = 0
    train_state = TrainState(step=start_step)

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            job_config.checkpoint.enable_checkpoint
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)

    # --- training loop ---
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    maybe_enable_memory_snapshot(job_config)
    maybe_enable_profiling(job_config)

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )

    # basic iterator over train dataloader
    # Each batch should contain already-preprocessed tensors from build_hf_data_loader
    #for micro_step, batch in enumerate(train_loader):
    while train_state.step < job_config.training.steps:
        train_state.step += 1

        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            batch = next(data_iterator)
        
        # unpack common fields expected by your graphs
        input_ids = batch["input_ids"].to(device, non_blocking=True) # check `pin_memory`; it starts the transfer and immediately move on to the next operation, overlapping computation and data transfer.
        labels = batch.get("labels", batch["input_ids"]).to(device, non_blocking=True)
        pixel_values = batch.get("pixel_values", batch["pixel_values"]).to(device, non_blocking=True)
        n_image = batch.get("n_image", batch["n_image"]).to(device, non_blocking=True)

        # TODO: enable_embed_batch
        enable_embed_batch = True if (job_config.training.seq_len >= 16384 and job_config.training.batch_size > 1) else False
        enable_embed_batch = False

        with torch.no_grad():
            # TODO: make embed function for Qwen.2.5 VL
            if 'llava' in model_name.lower():
                inputs_embeds = model.embed(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            n_image=n_image,
                            enable_embed_batch=enable_embed_batch)
            elif 'qwen' in model_name.lower():
                # logic for image_grid_thw
                # grid_t * grid_h * grid_w == pixel_values.shape[1]
                # grid_h, grid_w = job_config.training.img_width // 14, job_config.training.img_height // 14
                # hw = grid_h * grid_w
                # grid_t = n_image // hw
                # n_vis_tokens = n_image.squeeze(-1)
                # assert torch.all(n_vis_tokens % hw == 0), "per-sample tokens must be divisible by H*W"
                # grid_t = (n_vis_tokens // hw).to(torch.int32).unsqueeze(-1)  # (N,1)
                # # make per-sample columns for H and W
                # grid_h_col = torch.full_like(grid_t, grid_h)  # (N,1), int32
                # grid_w_col = torch.full_like(grid_t, grid_w)  # (N,1), int32
                # image_grid_thw = torch.cat([grid_t, grid_h_col, grid_w_col], dim=1).to(pixel_values.device)
                # logger.info(f"image_grid_thw: {image_grid_thw.shape}")

                image_grid_thw = batch.get("image_grid_thw").to(device, non_blocking=True)
                inputs_embeds = model.embed(input_ids=input_ids,
                                            pixel_values=pixel_values,
                                            image_grid_thw=image_grid_thw)
            else:
                inputs_embeds = model.embed(input_ids=input_ids,
                                            pixel_values=pixel_values)

        logger.info(f"[rank{dp_rank}] input_embeds: {inputs_embeds.shape}")
        
        position_ids = batch.get("position_ids", None)
        if position_ids is not None:
            position_ids = position_ids.to(device, non_blocking=True)

        # TODO zero_grad() here ?
        #optimizers.zero_grad()

        # Optional: redistribute inputs for TP if CP is off (as in your code)
        if parallel_dims.tp_enabled and (not parallel_dims.cp_enabled) and inputs_embeds is not None:
            if not (parallel_dims.pp_enabled and False):  # if not first stage etc., simplified
                inputs_embeds = distribute_tensor(inputs_embeds, world_mesh['tp'], placements=[Shard(1)]).to_local()

        # --- Context Parallel context ---
        optional_context_parallel_ctx = (
            utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[input_ids, inputs_embeds, labels, position_ids],
                cp_seq_dims=[1, 1, 1, 1],
                cp_no_restore_buffers={input_ids, inputs_embeds, labels, position_ids},
                cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled else None
        )

        # model_parts = [model]  # for grad clip below

        # --- forward/backward (PP vs non-PP) ---
        if parallel_dims.pp_enabled:
            # Placeholder PP schedule hook: your original uses pp_schedule.step(...).
            # Here we call the model directly; in your env, wire the pp_schedule just like originals.
            with train_context(optional_context_parallel_ctx):
                logits = model.language_model(inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False) if hasattr(model, "language_model") else model(input_ids=input_ids, use_cache=False)
                # CP hack parity
                if (labels + torch.tensor([100], device=labels.device)).sum() == 0:
                    labels[:, -2] = input_ids[:, -1]
                loss = loss_fn(logits, labels)
                del logits
                loss.backward()
        else:
            with train_context(optional_context_parallel_ctx):
                if hasattr(model, "language_model"): # Llava family
                    logits = model.language_model(inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False)
                elif hasattr(model, "model"): # others such as qwen
                    logger.info(f"Training starts at step {train_state.step + 1}")
                    logits = model(input_ids=input_ids,
                                    inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False)
                else:
                    logits = model(input_ids=input_ids, use_cache=False)
                # CP hack parity
                if (labels + torch.tensor([100], device=labels.device)).sum() == 0:
                    labels[:, -2] = input_ids[:, -1]
                loss = loss_fn(logits, labels)
                del logits
                loss.backward()

        # --- grad clip & step ---
        utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            job_config.training.max_norm,
            foreach=True,
            pp_mesh=world_mesh["pp"] if parallel_dims.pp_enabled else None,
        )
        checkpoint.maybe_wait_for_staging()
        optimizers.step()
        lr_schedulers.step()
        for opt in optimizers:  # clear grads
            opt.zero_grad(set_to_none=True)

        # --- logging / checkpoint ---
        if train_state.step % job_config.logging.log_interval == 0:
            logger.info(f"step {train_state.step:6d} | loss {color.yellow}{loss.item():.4f}{color.reset}")

        if job_config.checkpoint.interval > 0 and train_state.step % job_config.checkpoint.interval == 0:
            save_dir = Path(job_config.checkpoint.save_dir) / f"step-{train_state.step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            states = {"model": combine_model_parts_state(model_parts)}
            dcp.save_state_dict(states, storage_writer=dcp.FileSystemWriter(str(save_dir)))
            save_checkpoint_s3(states, train_state.step, str(save_dir))

        gc_handler.maybe_collect()

    logger.info("Training finished.")


if __name__ == "__main__":
    # Expect launch with torchrun + torchtitan's CLI that constructs JobConfig,
    # e.g., `torchrun --nproc_per_node=8 train.py --job.config_file my_job.yaml`
    config = JobConfig()
    config.parse_args()
    main(config)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
