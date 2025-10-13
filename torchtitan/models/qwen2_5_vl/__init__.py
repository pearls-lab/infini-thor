from torchtitan.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForActionPrediction
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from transformers import AutoConfig
from torchtitan.train_spec import register_train_spec, TrainSpec

from .parallelize_qwen2_5_vl import parallelize_qwen2_5_vl
#from .pipeline_qwen2_5_vl import pipeline_qwen2_5_vl

__all__ = [
    "parallelize_qwen2_5_vl",
    #"pipeline_qwen2_5_vl",
    "Qwen2_5_VLForActionPrediction",
    "qwen2_5_vl_configs",
]

qwen2_5_vl_configs = {
    # prob need to change variable names such as `dim`, `n_kv_heads` ...
    '7B': AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
}

register_train_spec(
    TrainSpec(
        name="Qwen/Qwen2.5-VL-7B-Instruct",
        cls=Qwen2_5_VLForActionPrediction,
        config=qwen2_5_vl_configs,
        parallelize_fn=parallelize_qwen2_5_vl,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)
