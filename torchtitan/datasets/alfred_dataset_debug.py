from torchtitan.datasets.processor import build_hf_processor
from huggingface_hub import snapshot_download
from torchtitan.logging import init_logger, logger

init_logger()

model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

processor = build_hf_processor(model_name)
tokenizer = processor.tokenizer
#img_data_dir = snapshot_download(repo_id="bosungkim/alfred-small-img", repo_type="dataset", allow_patterns="*.tar", local_dir='data/alfred-small-img')
#img_data_dir = img_data_dir = '/root/torchtitan/data/alfred-full/data'
traj_data_dir = '/data/bkim/alfred/data/full_2.1.0/train_new_traj'
img_data_dir = '/data/bkim/alfred/data/full_2.1.0/train_new_tar'
processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})

# from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.datasets.alfred_dataset_long_ctx import ALFREDDataset, AlfredDataLoader
dataset = ALFREDDataset(
    processor=processor,
    traj_data_dir=traj_data_dir,
    img_data_dir=img_data_dir,
    max_seq_len=262144)
dp_rank=0
data_loader = AlfredDataLoader(dp_rank, dataset,
                                batch_size=1,
                                world_size=1)

count = 0
for i, batch in enumerate(data_loader):
    print(i, batch['input_ids'].shape)
    count += 1
print(count)
    #print(batch['input_ids'])
