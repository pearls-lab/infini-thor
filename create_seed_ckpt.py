import os
import torch
import torch.nn as nn
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train_spec import get_train_spec
import torch.distributed.checkpoint as dcp


def main(job_config: JobConfig):
    init_logger()

    model_name = job_config.model.name
    train_spec = get_train_spec(model_name)
    model_cls = train_spec.cls
    model_config = train_spec.config[job_config.model.flavor]
    model_config.norm_type = job_config.model.norm_type

    model = model_cls.from_pretrained(model_name, config=model_config)

    output_dir = os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder, 'step-0')
    os.makedirs(output_dir, exist_ok=True)
    dcp.save({"model": model.state_dict()}, checkpoint_id=output_dir)
    logger.info(f"Created seed checkpoint: {output_dir}")
    return

if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()

    main(config)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
