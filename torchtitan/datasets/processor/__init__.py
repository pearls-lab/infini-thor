from transformers import AutoProcessor

def build_hf_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name)
