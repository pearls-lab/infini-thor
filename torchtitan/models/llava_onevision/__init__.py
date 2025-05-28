from torchtitan.models.llava_onevision.model import LlavaOnevisionForConditionalGeneration
from torchtitan.optimizer import build_lr_schedulers, build_llava_optimizers, build_optimizers
from transformers import AutoConfig
from torchtitan.train_spec import register_train_spec, TrainSpec

from torchtitan.models.llava_onevision.modeling_qwen2 import Qwen2Model
from torchtitan.models.llava_onevision.configuration_qwen2 import Qwen2Config

from .parallelize_llava import parallelize_llava
from .pipeline_llava import pipeline_llava

__all__ = [
    "parallelize_llava",
    "pipeline_llava",
    "LlavaOnevisionForConditionalGeneration",
    "llava_onevision_configs",
]

llava_onevision_configs = {
    # prob need to change variable names such as `dim`, `n_kv_heads` ...
    '7B': AutoConfig.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
}

register_train_spec(
    TrainSpec(
        name="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        cls=LlavaOnevisionForConditionalGeneration,
        config=llava_onevision_configs,
        parallelize_fn=parallelize_llava,
        pipelining_fn=pipeline_llava,
        build_optimizers_fn=build_llava_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)
