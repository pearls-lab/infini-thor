import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import distribute_module, distribute_tensor, DTensor, Replicate, Shard

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader, build_hf_processor
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_tokenizer
from torchtitan.parallelisms import ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.train_spec import get_train_spec
from torchtitan.utils import device_module, device_type, import_module_from_path

from huggingface_hub import HfApi, repo_exists

api = HfApi()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_global_rank():
    return int(os.environ.get("RANK", "0"))


def set_nested_attr(obj, name, value):
    """Register a buffer on nested modules (works through lists/ModuleLists)."""
    parts = name.split(".")
    if not hasattr(obj, parts[0]):
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


def upload_ckpt_hf(output_dir, repo_id, path_in_repo):
    """Upload a checkpoint folder to a HuggingFace model repo."""
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def warmup_dynamic_rope_scaling(model, device, seq_len, rope_kwargs):
    """Warm up RoPE scaling buffers to avoid on-the-fly reallocs during training."""
    try:
        layers = model.language_model.model.layers
        config = model.language_model.config if hasattr(model.language_model, "config") else model.config

        if rope_kwargs.get("rope_type") == "yarn":
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
    """Clip gradient norm for a mix of DTensors and regular tensors."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]

    local_norms = []
    for grad in grads:
        if isinstance(grad, DTensor):
            local_norm = grad.norm(p=norm_type).to_local()
            local_norms.append(local_norm)
        else:
            if norm_type == torch.inf:
                local_norm = grad.data.abs().max()
            else:
                local_norm = grad.data.norm(p=norm_type)
            local_norms.append(local_norm)

    if norm_type == torch.inf:
        total_norm = torch.stack(local_norms).max()
    else:
        total_norm = torch.sqrt(torch.stack([n**2 for n in local_norms]).sum())

    clip_coeff = max_norm / (total_norm + 1e-6)

    if clip_coeff < 1.0:
        for grad in grads:
            if isinstance(grad, DTensor):
                grad.mul_(clip_coeff)
            else:
                grad.data.mul_(clip_coeff)

    return total_norm


@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        import_module_from_path(job_config.experimental.custom_model_path)

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # --- Distributed setup ---
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

    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for MFU: {gpu_peak_flops:.3e}")
    logger.info(f"ParallelDims: {parallel_dims}")

    # --- Parallel meshes ---
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    dp_pg = world_mesh["dp"].get_group() if parallel_dims.dp_enabled else None
    tp_mesh = world_mesh["tp"] if parallel_dims.tp_enabled else None

    local_rank, global_rank = get_local_rank(), get_global_rank()

    # --- Model spec & config ---
    model_name = job_config.model.name
    train_spec = get_train_spec(model_name)
    model_cls = train_spec.cls
    model_config = train_spec.config[job_config.model.flavor]

    model_config.norm_type = job_config.model.norm_type
    text_config = getattr(model_config, "text_config", None) or model_config

    if job_config.training.rope_theta:
        text_config.rope_theta = job_config.training.rope_theta

    # Flash Attention 2 settings
    if job_config.training.attn_impl == "flash_attention_2":
        model_config.attn_impl = "flash_attention_2"
        text_config._attn_implementation = "flash_attention_2"
        text_config.use_sliding_window = True
        text_config.max_window_layers = 0

    # --- Tokenizer / processor / dataloader ---
    processor = build_hf_processor(model_name)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|act|>", "<|goal|>"]})

    data_loader = build_data_loader(
        job_config,
        processor,
        split="train",
        rank=global_rank,
        dp_world_size=dp_degree,
        dp_rank=dp_rank,
    )

    # --- Warm-up pass to capture buffers & RoPE scaling ---
    buffers_dict = None
    if "llava" in model_name.lower():
        model_tmp = model_cls.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        if job_config.training.rope_type:
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
        torch.cuda.empty_cache()

    # --- Meta init for placement with TP/PP/CP ---
    with torch.device("meta"):
        if "llava" in model_name.lower() or "qwen" in model_name.lower():
            model_config._attn_implementation = job_config.training.attn_impl
            model = model_cls(model_config)
        else:
            model = model_cls.from_model_args(model_config)

    model_param_count = utils.get_num_params(model)
    logger.info(f"Building {train_spec.name} {job_config.model.flavor} with {model_config}")
    logger.info(f"Model {model_name} size: {model_param_count:,} parameters")

    gradient_accumulation_steps = job_config.training.gradient_accumulation_steps

    def loss_fn(pred, labels):
        loss = torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        return loss

    # --- Distribute model (PP/TP) ---
    model_parts = [model]

    if parallel_dims.pp_enabled:
        placements = [Replicate()]
        if parallel_dims.tp_enabled and tp_mesh is not None:
            placements = [Shard(0)]
        model = distribute_module(
            model,
            world_mesh["tp"] if parallel_dims.tp_enabled else world_mesh["dp"],
            placements=placements,
        )
        model_parts = [model]
    elif parallel_dims.tp_enabled:
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=device_type)
        if buffers_dict:
            with torch.no_grad():
                for name, buf in buffers_dict.items():
                    set_nested_attr(model, name, buf.to(device_type))
        model.to(dtype=torch.bfloat16)
        model.train()
        model_parts = [model]
    else:
        model.to_empty(device=device_type)
        if buffers_dict:
            with torch.no_grad():
                for name, buf in buffers_dict.items():
                    set_nested_attr(model, name, buf.to(device_type))
        model.to(dtype=torch.bfloat16)
        model.train()
        model_parts = [model]

    # --- Optimizer / scheduler / checkpoint ---
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)

    train_state = TrainState()

    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )
    checkpoint.load(step=job_config.checkpoint.load_step)

    # --- Training loop setup ---
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # Replay logged losses to TensorBoard (from checkpoint)
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

    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"gradient accumulation steps {gradient_accumulation_steps}, "
        f"global batch size {job_config.training.batch_size * dp_degree * gradient_accumulation_steps}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )

    in_ids_lengths = []
    in_embeds_lengths = []

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

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16, non_blocking=True)
            n_image = batch["n_image"].to(device, non_blocking=True)

            # Global sample counting for distributed logging
            local_samples = torch.tensor([input_ids.size(0)], device=device, dtype=torch.int64)
            if parallel_dims.dp_replicate_enabled or parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                if dp_pg is not None:
                    dist.all_reduce(local_samples, op=dist.ReduceOp.SUM, group=dp_pg)

            logger.info(
                f"step: {train_state.step} input_ids: {input_ids.shape}, "
                f"pixel_values: {pixel_values.shape}, "
                f"image_grid_thw: {batch.get('image_grid_thw', torch.tensor([])).shape}"
            )
            
            # Qwen2.5-VL uses image_grid_thw for RoPE; LLaVA does not (even though its name contains "qwen2")
            is_qwen_vl = "qwen" in model_name.lower() and "llava" not in model_name.lower()
            with torch.no_grad():
                if is_qwen_vl:
                    image_grid_thw = batch.get("image_grid_thw").to(device, non_blocking=True)
                    inputs_embeds = model.embed(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                    position_ids, rope_deltas = model.get_rope_index(
                        input_ids, image_grid_thw, None, None, None,
                    )
                    model.rope_deltas = rope_deltas
                else:
                    inputs_embeds = model.embed(input_ids=input_ids, pixel_values=pixel_values, n_image=n_image)
                    position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

            in_ids_lengths.append(input_ids.shape[1])
            in_embeds_lengths.append(inputs_embeds.shape[1])

            # Redistribute inputs for TP when CP is off
            if parallel_dims.tp_enabled and not parallel_dims.cp_enabled and inputs_embeds is not None:
                inputs_embeds = distribute_tensor(inputs_embeds, world_mesh["tp"], placements=[Shard(1)]).to_local()

            # --- Context Parallel ---
            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, inputs_embeds, labels, position_ids],
                    cp_seq_dims=[1, 1, 1, 2],
                    cp_no_restore_buffers={input_ids, inputs_embeds, labels, position_ids},
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            # --- Forward / backward ---
            if parallel_dims.pp_enabled:
                with train_context(optional_context_parallel_ctx):
                    logits = (
                        model.language_model(inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False)
                        if hasattr(model, "language_model")
                        else model(input_ids=input_ids, use_cache=False)
                    )
                    if (labels + torch.tensor([100], device=labels.device)).sum() == 0:
                        labels[:, -2] = input_ids[:, -1]
                    loss = loss_fn(logits, labels)
                    del logits
                    loss.backward()
            else:
                with train_context(optional_context_parallel_ctx):
                    if parallel_dims.cp_enabled:
                        output = model(
                            inputs_embeds=inputs_embeds,
                            position_ids=position_ids,
                            use_cache=False,
                        )
                    else:
                        output = model(
                            inputs_embeds=inputs_embeds,
                            use_cache=False,
                        )
                    logits = output if isinstance(output, torch.Tensor) else output.logits

                    if (labels + torch.tensor([100], device=labels.device)).sum() == 0:
                        labels[:, -2] = input_ids[:, -1]
                    loss = loss_fn(logits, labels)
                    del logits
                    loss.backward()

            accumulated_losses.append(loss.detach().clone())

        # --- Gradient clipping & optimizer step ---
        dtensor_safe_clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            job_config.training.max_norm,
        )

        checkpoint.maybe_wait_for_staging()
        optimizers.step()
        lr_schedulers.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # --- Logging ---
        if train_state.step % job_config.metrics.log_freq == 0:
            if (
                parallel_dims.dp_replicate_enabled
                or parallel_dims.dp_shard_enabled
                or parallel_dims.cp_enabled
            ):
                loss = torch.sum(torch.stack(accumulated_losses))
                global_avg_loss, global_max_loss = (
                    utils.dist_mean(loss, world_mesh["dp_cp"]),
                    utils.dist_max(loss, world_mesh["dp_cp"]),
                )
            else:
                global_avg_loss = global_max_loss = loss.item()

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
            )

        # --- Checkpointing ---
        if job_config.checkpoint.interval > 0 and train_state.step % job_config.checkpoint.interval == 0:
            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.checkpoint.interval)
            )
            dist.barrier()
            if local_rank == 0 and job_config.job.hf_repo_id:
                upload_ckpt_hf(
                    Path(checkpoint.folder) / f"step-{train_state.step}",
                    job_config.job.hf_repo_id,
                    f"step-{train_state.step}",
                )
            dist.barrier()

    logger.info(f"avg input_ids length: {np.array(in_ids_lengths).mean():.1f}")
    logger.info(f"avg input_embeds length: {np.array(in_embeds_lengths).mean():.1f}")

    # Final checkpoint
    checkpoint.save(train_state.step, force=True)
    dist.barrier()
    if local_rank == 0 and job_config.job.hf_repo_id:
        upload_ckpt_hf(
            Path(checkpoint.folder) / f"step-{train_state.step}",
            job_config.job.hf_repo_id,
            f"step-{train_state.step}",
        )
    dist.barrier()
    logger.info("Training finished.")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()

    if config.job.hf_repo_id:
        if not repo_exists(config.job.hf_repo_id, repo_type="model"):
            raise ValueError(f"Invalid Hugging Face repo ID: {config.job.hf_repo_id}")

    main(config)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
