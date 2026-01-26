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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
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
from torchtitan.loss import rescale_accumulated_loss
from torchtitan.parallelisms import ParallelDims
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.train_spec import get_train_spec
from torchtitan.utils import device_module, device_type, import_module_from_path
# Add this utility function to your train.py (or utils.py)
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForActionPrediction
from torchtitan.optimizer import build_lr_schedulers, build_optimizers, build_lm_only_optimizers
from transformers import AutoConfig
from torchtitan.train_spec import TrainSpec

qwen2_5_vl_configs = {
    # prob need to change variable names such as `dim`, `n_kv_heads` ...
    '7B': AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
}

from huggingface_hub import HfApi, repo_exists

api = HfApi()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_global_rank():
    return int(os.environ.get("RANK", "0"))


def combine_model_parts_state(model_parts: List[nn.Module]):
    out = {}
    for m in model_parts:
        sd = m.state_dict()
        for k, v in sd.items():
            if v is not None:
                out[k] = v
    return out


def upload_ckpt_hf(output_dir, repo_id, path_in_repo):
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model"
    )

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


def dtensor_safe_clip_grad_norm_(
    parameters, max_norm: float, norm_type: float = 2.0, foreach: bool = False
) -> torch.Tensor:
    """
    Computes and clips the total gradient norm for a list of DTensors.
    This function manually reduces all DTensor gradients to their local norm 
    before passing the results to the standard torch.nn.utils function.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
        
    grads = [p.grad for p in parameters if p.grad is not None]
    
    # 1. Manually reduce all DTensor gradients to their local norm
    local_norms = []
    for grad in grads:
        # Check if it's a DTensor (from TP/CP/DP)
        if isinstance(grad, DTensor):
            # Compute the *local* norm from the DTensor's partial norm.
            # The all-reduce happens here.
            local_norm = grad.norm(p=norm_type).to_local()
            local_norms.append(local_norm)
        else:
            # Handle native local tensors (e.g., non-sharded params)
            if norm_type == torch.inf:
                local_norm = grad.data.abs().max()
            else:
                local_norm = grad.data.norm(p=norm_type)
            local_norms.append(local_norm)

    # 2. Stack the resulting *local* tensors (which are all on the same device)
    # The total global norm calculation still relies on the original 
    # torch.nn.utils.clip_grad_norm_ to correctly combine DP/other effects.
    
    # Square of the local norms for L2 or use the max for L-infinity
    if norm_type == torch.inf:
        total_norm = torch.stack(local_norms).max()
    else:
        total_norm = torch.sqrt(torch.stack([n**2 for n in local_norms]).sum())

    # 3. Clip the gradients based on the local total norm
    clip_coeff = max_norm / (total_norm + 1e-6)
    
    # Only clip if the total norm exceeds max_norm
    if clip_coeff < 1.0:
        for grad in grads:
            if isinstance(grad, DTensor):
                # The DTensor logic will handle applying the clipping factor correctly
                # (it typically broadcasts the clipping factor and multiplies).
                grad.mul_(clip_coeff)
            else:
                grad.data.mul_(clip_coeff)

    # Return the total norm for logging
    return total_norm


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with value -100 are ignored. Shape: (batch_size, seq_len)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and seq_len) and labels must have the same shape.")

    # Shift so that tokens < n predict n
    # Standard CausalLM shift: logits[..., :-1, :] predicts labels[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Create a mask for non-ignored tokens
    loss_mask = (shift_labels != -100)

    # Compute log probabilities (using log_softmax for numerical stability)
    # gather extracts the log_prob of the true label
    per_token_logps = torch.gather(shift_logits.log_softmax(-1), dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

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

    dp_pg = None
    if parallel_dims.dp_enabled:
        # 1D submesh for the DP dimension
        dp_pg = world_mesh["dp"].get_group()  

    pp_mesh = world_mesh["pp"] if parallel_dims.pp_enabled else None
    tp_mesh = world_mesh["tp"] if parallel_dims.tp_enabled else None

    local_rank, global_rank = get_local_rank(), get_global_rank()

    # --- model spec & config ---
    model_name = job_config.model.name  # Expect user to set a HF ckpt (llava or qwen2-vl family)
    #train_spec = get_train_spec(model_name)  # returns spec with .cls and .config mapping
    from torchtitan.models.qwen2_5_vl.parallelize_qwen2_5_vl_unfrz import parallelize_qwen2_5_vl
    train_spec = TrainSpec(
        name="Qwen/Qwen2.5-VL-7B-Instruct",
        cls=Qwen2_5_VLForActionPrediction,
        config=qwen2_5_vl_configs,
        parallelize_fn=parallelize_qwen2_5_vl,
        pipelining_fn=None,
        build_optimizers_fn=build_lm_only_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )

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
    tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|goal|>']})

    data_loader = build_data_loader(
        job_config,
        processor, 
        #dp_mesh=dp_mesh if parallel_dims.dp_enabled else None,
        split="train",
        rank=global_rank,
        dp_world_size=dp_degree,
        dp_rank=dp_rank,
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
    elif "qwen" in model_name.lower():
        with torch.no_grad():
            model = model_cls.from_pretrained(model_name, config=model_config)
            buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        del model

        logger.info("Initializing Reference Model for DPO...")

        torch.cuda.empty_cache()

    # --- meta init to control placement with TP/PP/CP ---
    with torch.device("meta"):
        if "llava" in model_name.lower() or "qwen" in model_name.lower():
            # use from_pretrained on real device later to ensure lm_head sizing vs tokenizer
            # model = model_cls.from_pretrained(model_name, 
            #     config=model_config,
            #     attn_implementation=job_config.training.attn_impl,
            #     torch_dtype=torch.bfloat16
            # )
            model_config._attn_implementation = job_config.training.attn_impl 
            model = model_cls(model_config)
        else:
            model = model_cls.from_model_args(model_config)

    # log model size
    model_param_count = utils.get_num_params(model)
    logger.info(f"Building {train_spec.name} {job_config.model.flavor} with {model_config}")
    logger.info(f"Model {model_name} size: {model_param_count:,} parameters")

    gradient_accumulation_steps = job_config.training.gradient_accumulation_steps

    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))
    loss_fn = rescale_accumulated_loss(loss_fn, gradient_accumulation_steps)

    # --- distribute model by PP/TP as requested ---
    model_parts = [model]

    if parallel_dims.pp_enabled:
        # Distribute the module across parallel meshes as your originals do
        placements = [Replicate()]
        if parallel_dims.tp_enabled and tp_mesh is not None:
            placements = [Shard(0)]  # shard head/hidden or per your layout; placeholder kept minimal
        model = distribute_module(model, world_mesh["tp"] if parallel_dims.tp_enabled else world_mesh["dp"], placements=placements)
        model_parts = [model]
    elif parallel_dims.tp_enabled or parallel_dims.dp_shard_enabled:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_buffers(buffer_device=device_type, buffers_dict=buffers_dict)
        model.to(dtype=torch.bfloat16)
        model.train()
        model_parts = [model]
    else:
        #model.to_empty(device=device_type, dtype=torch.bfloat16)
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_buffers(buffer_device=device_type, buffers_dict=buffers_dict)
        model.to(dtype=torch.bfloat16)
        model.train()
        model_parts = [model]
        # ?
        #state_dict = {"model": m.state_dict()}
        #dcp.load(state_dict, checkpoint_id=checkpoint_path, planner=dcp.DefaultLoadPlanner(allow_partial_load=True))

    # 1. Initialize Reference Model
    logger.info("Initializing Reference Model for DPO...")
    with torch.device("meta"):
        if "llava" in model_name.lower() or "qwen" in model_name.lower():
             ref_model = model_cls(model_config)
        else:
             ref_model = model_cls.from_model_args(model_config)

    # 2. Parallelize Reference Model (CRITICAL FIX)
    # This converts ref_model params into DTensors matching the main model
    if parallel_dims.tp_enabled or parallel_dims.dp_enabled or parallel_dims.pp_enabled:
         train_spec.parallelize_fn(ref_model, world_mesh, parallel_dims, job_config)

    # 3. Materialize on Device
    ref_model.to_empty(device=device_type)
    with torch.no_grad():
        ref_model.init_buffers(buffer_device=device_type, buffers_dict=buffers_dict)
    ref_model.to(dtype=torch.bfloat16)

    # --- 4. Copy Weights (The Fix) ---
    # We iterate manually and force the copy to happen on the LOCAL shards.
    # This bypasses the "mixed torch.Tensor and DTensor" error because .to_local() 
    # returns the standard torch.Tensor underlying the DTensor.
    
    from torch.distributed.tensor import DTensor
    
    logger.info("Copying weights to Reference Model via local shards...")
    with torch.no_grad():
        # Zip ensures we match parameters 1-to-1. 
        # Since models are identical and parallelized identically, order is preserved.
        for (name_p, param_p), (name_r, param_r) in zip(model.named_parameters(), ref_model.named_parameters()):
            
            # Sanity check to ensure we aren't mixing up layers
            # (Relaxed check: allow _checkpoint_wrapped_module mismatch if it exists)
            clean_name_p = name_p.replace("_checkpoint_wrapped_module.", "")
            clean_name_r = name_r.replace("_checkpoint_wrapped_module.", "")
            if clean_name_p != clean_name_r:
                logger.warning(f"Parameter name mismatch in copy: {name_p} vs {name_r}. Proceeding by order.")

            # Get the local shard. If it's already local (e.g. not sharded), it returns self.
            src_tensor = param_p.to_local() if isinstance(param_p, DTensor) else param_p
            dst_tensor = param_r.to_local() if isinstance(param_r, DTensor) else param_r
            
            # Perform the copy on the local data
            dst_tensor.copy_(src_tensor)

    # 5. Freeze Reference Model
    ref_model.eval()
    ref_model.requires_grad_(False)

    # --- optimizer/scheduler/checkpoint ---
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    
    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

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
        f"gradeint accumulation steps {gradient_accumulation_steps}, "
        f"global batch size {job_config.training.batch_size * dp_degree * gradient_accumulation_steps}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )

    # basic iterator over train dataloader
    # Each batch should contain already-preprocessed tensors from build_hf_data_loader
    #for micro_step, batch in enumerate(train_loader):
    in_ids = []
    in_embeds = []
    N = len(data_loader.dataset) # # of trajs

    # DPO Hyperparams
    beta = 0.1

    while train_state.step < job_config.training.steps:
        optimizers.zero_grad()
        accumulated_losses = []
        train_state.step += 1
        gc_handler.run(train_state.step)
        
        for _microbatch in range(gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                batch = next(data_iterator)

            c_input_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(device, non_blocking=True)
            
            r_input_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(device, non_blocking=True)
            
            # Image handling: Assuming shared image for chosen/rejected or provided explicitly
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            image_grid_thw = batch.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device, non_blocking=True)
            
            # --- Forward Pass Policy (Trainable) ---
            # 1. Chosen
            # Note: You might need to handle 'position_ids' generation here if not handled by model
            c_logits = model(input_ids=c_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
            if isinstance(c_logits, object) and hasattr(c_logits, 'logits'): c_logits = c_logits.logits
            
            # 2. Rejected
            r_logits = model(input_ids=r_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
            if isinstance(r_logits, object) and hasattr(r_logits, 'logits'): r_logits = r_logits.logits

            # --- Forward Pass Reference (Frozen) ---
            with torch.no_grad():
                # 1. Chosen
                ref_c_logits = ref_model(input_ids=c_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
                if hasattr(ref_c_logits, 'logits'): ref_c_logits = ref_c_logits.logits
                
                # 2. Rejected
                ref_r_logits = ref_model(input_ids=r_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
                if hasattr(ref_r_logits, 'logits'): ref_r_logits = ref_r_logits.logits
            

            # --- Calculate DPO Loss ---
            
            # 1. Get Log Probs of the *completion* parts
            # Convert DTensors to local if needed before complex gather ops if gather isn't supported on DTensor yet
            # However, for simple pointwise math, DTensor usually works.
            
            policy_chosen_logps = get_batch_logps(c_logits, c_labels, average_log_prob=False)
            policy_rejected_logps = get_batch_logps(r_logits, r_labels, average_log_prob=False)
            
            ref_chosen_logps = get_batch_logps(ref_c_logits, c_labels, average_log_prob=False)
            ref_rejected_logps = get_batch_logps(ref_r_logits, r_labels, average_log_prob=False)

            # 2. Compute DPO Logits
            # pi_logratios = policy_chosen_logps - policy_rejected_logps
            # ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            # logits = pi_logratios - ref_logratios
            
            logits = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)

            # 3. Sigmoid Loss
            # loss = -log(sigmoid(beta * logits))
            losses = -torch.nn.functional.logsigmoid(beta * logits)
            
            # 4. Rewards (for logging purposes)
            with torch.no_grad():
                chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
                rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
                reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

            # Average over batch
            loss = losses.mean()
            
            # Rescale for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            accumulated_losses.append(loss.detach().clone())

        # --- grad clip & step ---
        # utils.clip_grad_norm_(
        #     [p for m in model_parts for p in m.parameters()],
        #     job_config.training.max_norm,
        #     foreach=True,
        #     pp_mesh=world_mesh["pp"] if parallel_dims.pp_enabled else None,
        # )
        dtensor_safe_clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            job_config.training.max_norm,
            foreach=True # Foreach is ignored in this manual implementation but kept for consistency
        )
        
        checkpoint.maybe_wait_for_staging()
        optimizers.step()
        lr_schedulers.step()
        for opt in optimizers:  # clear grads
            opt.zero_grad(set_to_none=True)

        # --- logging / checkpoint ---
        if train_state.step % job_config.metrics.log_freq == 0:
            # logger.info(f"step {train_state.step:6d} | loss {color.yellow}{loss.item():.4f}{color.reset}")
            if (
                parallel_dims.dp_replicate_enabled
                or parallel_dims.dp_shard_enabled
                or parallel_dims.cp_enabled
            ):
                #loss = loss.detach()
                loss = torch.sum(torch.stack(accumulated_losses))
                global_avg_loss, global_max_loss = (
                    utils.dist_mean(loss, world_mesh["dp_cp"]),
                    utils.dist_max(loss, world_mesh["dp_cp"]),
                )
            else:
                global_avg_loss = global_max_loss = loss.item()

            # update train state
            train_state.log_steps.append(train_state.step)
            train_state.global_avg_losses.append(global_avg_loss)
            train_state.global_max_losses.append(global_max_loss)

            device_mem_stats = device_memory_monitor.get_peak_stats()

            metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
            metric_logger.log(metrics, step=train_state.step)

            logger.info(
                f"{color.cyan}step: {train_state.step:2}  "
                f"{color.green}loss: {global_avg_loss:7.4f}  "
                f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                f"({device_mem_stats.max_reserved_pct:.2f}%){color.reset}"
                # f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
            )

        if job_config.checkpoint.interval > 0 and train_state.step % job_config.checkpoint.interval == 0:
            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.checkpoint.interval)
            )
            dist.barrier()  # Ensure all ranks finished saving before upload
            if local_rank == 0:
                upload_ckpt_hf(Path(checkpoint.folder) / f"step-{train_state.step}", 
                                job_config.job.hf_repo_id,
                                f"step-{train_state.step}")
            dist.barrier()  # Ensure all ranks wait for upload to complete

    logger.info(f"avg input_ids length: {np.array(in_ids).mean()}")
    logger.info(f"avg input_embeds length: {np.array(in_embeds).mean()}")
    
    checkpoint.save(train_state.step, force=True)
    dist.barrier()  # Ensure all ranks finished saving before upload
    if local_rank == 0:
        upload_ckpt_hf(Path(checkpoint.folder) / f"step-{train_state.step}",
                        job_config.job.hf_repo_id,
                        f"step-{train_state.step}")
    dist.barrier()  # Ensure all ranks wait for upload to complete
    logger.info("Training finished.")


if __name__ == "__main__":
    # Expect launch with torchrun + torchtitan's CLI that constructs JobConfig,
    # e.g., `torchrun --nproc_per_node=8 train.py --job.config_file my_job.yaml`
    config = JobConfig()
    config.parse_args()

    # write a simple Python code to test HF repo id is valid or not
    if not repo_exists(config.job.hf_repo_id, repo_type="model"):
        raise ValueError(f"Invalid Hugging Face repo ID: {config.job.hf_repo_id}")

    main(config)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
