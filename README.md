# $\infty$-THOR: Beyond Needle(s) in the Embodied Haystack

We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI.

$\infty$-THOR provides:

(1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories;

(2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents’ long-context reasoning ability; and 

(3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences.

To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.


<p align="center">
| <a href="https://arxiv.org/pdf/2505.16928"><b>Paper</b></a> | <a href="https://huggingface.co/datasets/PEARLS-Lab/infini-thor-nieh"> <b>Dataset</b> </a> | <a href="https://x.com/rajammanabrolu/status/1925945858664604025"> <b>Tweet</b> </a> |
</p>


# Setup

We provide a Dockerfile for setting up the environment. To build the image:
```
docker build -t infini-thor -f Dockerfile .
```
or pull image from the hub
```
docker pull bosung17/infini-thor
```


Then clone the code and install packages:

```bash
git clone https://github.com/pearls-lab/infini-thor.git
cd infini-thor
```

If you're not using Docker, need to install packages
```
pip install -r requirements.txt
```

Note: We highly recommend using FlashAttention 2 for faster training and evaluation. Use the following command to install:
```
pip install --no-build-isolation flash-attn
```

# Static Evaluation: Needle(s) in the Embodied Haystack (NiEH)

### Download and unzip QA Data
Download the NiEH set from huggingface dataset hub (you may need to set HF_TOKEN or login with `huggingface-cli login`)
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download PEARLS-Lab/infini-thor-nieh --repo-type dataset --local-dir /path/to/directory
```

Unzip metadata
```
tar xvf metadata.tar
```

---

### Data Format

**NiEH Data File (CSV)**:
- `qa_set_nieh_single_clue.csv`: Single-evidence QA set (Needle in the Embodied Haystack task)
- `qa_set_nieh_multi_clue.csv`: Multi-evidence QA set (Needle**s** in the Embodied Haystack task)

Each CSV file should contain the following columns:
- `traj_id`: Trajectory identifier
- `question`: The question to be answered
- `gt_img_idx`: List of ground truth image indices
- `answer`: List of acceptable answers


We also need GT images and metadata to build embodied haystacks. The
**metadata directory structure** is:
```
metadata/
├── traj_id/
    ├── img/
    │   ├── *.png (image files)
    ├── metadata.json
    ├── traj.txt
    ├── expert_log.json
```

---

### Run evaluation

The evaluation script supports multiple modes via `--eval_mode`:

| Mode | Description |
|------|-------------|
| `full_traj` | Feed the entire trajectory image sequence (default with `--full_traj`) |
| `haystack` | Build a controlled haystack context at varying needle depths (default without `--full_traj`) |
| `clip_retrieval` | Retrieve top-K images via CLIP similarity before prompting |
| `truncate_head` | Keep only the tail of the trajectory that fits in `--ctx_size` |
| `interleaved` | Interleave state images with action text from trajectory data |
| `text_state` | Use a text state summary + last frame |
| `video` | Pass the trajectory as a video file |

---

**Full trajectory evaluation** — the model receives the entire trajectory as input:

```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --full_traj
```

QA performance with the full trajectory as input:

| Model | Single-Evidence | Multi-Evidence |
|-------|----------------|----------------|
| LLaVA-OV (7B) | 0% | 0% |
| DeepSeek-VL (7B) | 0% | 0% |
| Qwen2.5-VL (7B) | 47.35% | 36.6% |
| Gemini 2.0 Flash | 67.36% | 30.94% |


*LLaVA-OV and DeepSeek-VL fail to handle long contexts beyond their pretraining limits*

Note: To run the DeepSeek-VL model, follow the instruction [here](https://github.com/deepseek-ai/DeepSeek-VL).

---

**Haystack evaluation** — build a controlled context at varying needle depths with a given context size (e.g., `--ctx_size 256` means 256K tokens):

```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --ctx_size 256
```

---

**Context extension** — apply RoPE scaling (e.g., YaRN) to extend the model's effective context:

```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --ctx_size 256 \
    --ctx_extension yarn \
    --ctx_extension_factor 4.0
```

---

**Interleaved evaluation** — interleave state images with action text (requires `--traj_dir`):

```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --full_traj --eval_mode interleaved \
    --traj_dir path/to/traj_jsons
```

---

**Evaluate a local checkpoint** — use `--base_model` to specify the base architecture:

```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name_or_path path/to/local/checkpoint \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --full_traj
```

**Additional flags:**
- `--n_img_token <int>`: Override the per-image token count (auto-detected for known models)
- `--attn_impl {flash_attention_2,sdpa,eager}`: Attention implementation (default: `flash_attention_2`)
- `--clip_model_name <name>`: CLIP model for `clip_retrieval` mode (default: `openai/clip-vit-large-patch14`)
- `--clip_top_k <int>`: Number of top images to retrieve with CLIP (default: 10)

## Interactive Evaluation

Interactive evaluation works with the [AI2THOR](https://ai2thor.allenai.org) simulator.
Our dataset is built using an older version of AI2THOR (v2.0.1), which requires Python 3.6 to run properly.
We recommend using the provided Dockerfile to avoid compatibility issues related to Python version and rendering.

Alternative way is installing Python 3.6 env manually (for non-docker user only):

```
conda create -y -n ai2thor_env python=3.6
conda activate ai2thor_env && \
pip install --ignore-installed ai2thor==2.1.0 flask requests opencv-python-headless==4.5.3.56 pillow
```

**Download checkpoints**

```
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='PEARLS-Lab/infini-thor', allow_patterns=['checkpoints/**'], local_dir='.', resume_download=True)"
```

**Running X server**

AI2THOR uses Unity3D to render scenes, which requires a graphical environment.
Since most GPU servers and containers run headlessly, X server must be manually started to simulate a display.
Use the script below to start a virtual X server on display 0:
```
# use tmux or run in background
python env_utils/startx.py 0
```

**Running AI2THOR service**

We use a microservice to solve the version compatibility issue between AI2THOR 2.1.0 (python 3.6) and PyTorch (python 3.10 or later) environments.
`ai2thor_service.py` runs the AI2THOR simulator, provides a REST API for environment interactions and handles all scene management and agent actions.
This works over the Flask and let us communicate between the simulator and agent over TCP.

```
# use tmux or run in background
conda activate ai2thor_env
python env_utils/ai2thor_service.py
```

Running the evaluation (need to deactivate `ai2thor_env` env if needed):

```
export MODEL_LABEL=llava_onevison_7b_32k
python run_interactive_eval.py \
  --checkpoint checkpoints/$MODEL_LABEL \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --flash_attn
```

# Generating $\infty$ trajectories
1. Start X Server
```
# use tmux or run in background
python env_utils/startx.py 0
```

2. Run the trajectory generation script
```
conda activate ai2thor_env
cd env_utils
python generate_traj.py --min_step 500
```
`--min_step`: Minimum number of steps required for each trajectory; the script continues running until it generates trajectories meeting the minimum step requirement

`--testset`: Test examples include the synthetic task at the end of the trajectory. Run an additional loop to create final synthetic tasks. Use this flag to generate valid or test sets. 

`--scene_ids`: Comma-separated floor plans to generate for (e.g. `--scene_ids 230,210`). By default the script sweeps every scene in `constants.SCENE_TYPE`.

`--num_traj_per_scene`: How many trajectories to generate per floor plan (default 1). Note that each attempt (successful or not) consumes one slot.

`--seed`: Random seed. Useful when running several workers in parallel so they explore different task sequences.

`--max_fail`: How many *executed-then-failed* plans to tolerate before rolling back the last subgoal (default 20). Unsatisfiable task samples and planner failures no longer consume this budget — they retry freely under a separate generous cap.

`--save_floor`: If set (e.g. `--save_floor 700`), an attempt that gives up — or whose final validation replay fails — rolls back to the longest prefix that replays cleanly and saves it if it has at least this many steps, instead of discarding the whole episode. Recommended for long-horizon generation.

`--replay_every`: Run the full validation replay every N accepted subgoals instead of after every one (default 1). Replays cost O(episode length), so this substantially speeds up long trajectories; the episode is always fully replayed before saving.

`--no_scene_goal_filter`: By default only goal types marked achievable for the scene's room type in `constants.GOALS_VALID` are sampled (e.g. no `pick_heat_then_place_in_recep` in a living room). Pass this flag to restore unfiltered sampling over all 7 goal types.

Each saved trajectory embeds a `gen_info` block (generation settings, per-subgoal RNG seeds, git revision) and the script maintains a `status.json` in the save directory for monitoring. If the AI2THOR/Unity process crashes mid-run, the controller is restarted automatically and generation resumes from the last accepted state.

Output: Generated trajectories are saved to the `new_trajectories/` directory.

### Running several generation workers in parallel

Generation is CPU-bound and one worker uses a single GPU lightly, so it is usually worth running
several at once. Give each worker its own X screen, its own save path, and its own planner scratch
dir (`INFINI_LOG_DIR`, otherwise workers overwrite each other's PDDL problem files):

```bash
for i in 0 1 2 3; do
  INFINI_LOG_DIR=/tmp/gen_$i \
  python generate_traj.py --scene_ids 230 --min_step 2000 --min_subgoal 40 \
      --seed $((i+1)) --x_display 0.$i --save_path new_trajectories_$i &
done
```

Longer trajectories take super-linearly longer to generate: after every subgoal the whole episode is
replayed from step 0 to verify it still reproduces, so a 2,000-step trajectory costs far more than
4x a 500-step one.

### Rendering a trajectory back into frames

Generation stores action plans only — no images — so that the search is not slowed down by
rendering. `render_traj.py` deterministically replays a saved trajectory and writes one frame per
low-level action plus a `timeline.json` with the per-step action, subgoal and object observations:

```bash
python env_utils/render_traj.py \
    --traj_json new_trajectories/floorplan230/floorplan230_58_2043_*.json \
    --out_dir   render/fp230 \
    --width 960 --height 540 --quality Ultra
```

`docs/` hosts the [project website](https://pearls-lab.github.io/infini-thor). To rebuild the
trajectory it plays (video, poster, slit-scan strip and `_data/trajectories.json`) from one or more
render directories:

```bash
python scripts/build_web_traj_assets.py \
    --hero render/fp230 --gallery render/fp210 render/fp323
```

# Training

We provide a distributed training script built on [torchtitan](https://github.com/pytorch/torchtitan) that supports Tensor Parallelism (TP), Data Parallelism (DP), and Context Parallelism (CP).

### Download training data

Download the training set from the HuggingFace dataset hub:
```bash
huggingface-cli download PEARLS-Lab/infini-thor --repo-type dataset --local-dir /path/to/infini-thor-data
```

### Config files

Pre-built config files are provided in `configs/`:

**LLaVA-OneVision 7B**

| Config | Parallelism | Sequence Length | Use Case |
|--------|-------------|-----------------|----------|
| `ft_llava_ov_7B_tp4_dp2.toml` | TP4 x DP2 | 32K | Default 8-GPU setup |
| `ft_llava_ov_7B_tp2_dp4.toml` | TP2 x DP4 | 32K | Higher data throughput |
| `ft_llava_ov_7B_tp2_cp2_dp2.toml` | TP2 x CP2 x DP2 | 64K | Long-context training |

**Qwen2.5-VL 7B**

| Config | Parallelism | Sequence Length | Use Case |
|--------|-------------|-----------------|----------|
| `ft_qwen_25vl_7B_infini_tp4_dp2.toml` | TP4 x FSDP | 42K | Default multi-GPU setup |
| `ft_qwen_25vl_7B_infini_cp8.toml` | CP8 x FSDP | 32K | 8-way context parallelism |
| `ft_qwen_25vl_7B_infini_cp16.toml` | CP16 x FSDP | 128K | 16-way context parallelism for very long sequences |

### Create seed checkpoint

Before training, create a seed checkpoint that converts the pretrained model weights into the distributed checkpoint format:

```bash
export CONFIG_FILE=./configs/ft_llava_ov_7B_tp4_dp2.toml
torchrun --nproc_per_node 1 \
    create_seed_ckpt.py --job.config_file $CONFIG_FILE
```

This saves the initial checkpoint to `{dump_folder}/{checkpoint.folder}/step-0/`.

### Run training

```bash
torchrun --nproc_per_node 8 \
    --local-ranks-filter 0 \
    train.py --job.config_file $CONFIG_FILE \
    --training.traj_data_dir /path/to/infini-thor-data/train/train_traj \
    --training.img_data_dir /path/to/infini-thor-data/train/img_tar
```

To upload checkpoints to HuggingFace Hub during training, add:
```bash
    --job.hf_repo_id your-username/your-repo-name
```

### Key training options

All options can be set in the TOML config file or overridden via command line (`--section.key value`):

| Option | Description | Default |
|--------|-------------|---------|
| `--training.seq_len` | Max sequence length | 32768 |
| `--training.batch_size` | Per-GPU batch size | 1 |
| `--training.gradient_accumulation_steps` | Gradient accumulation steps | 4 |
| `--training.steps` | Total training steps | 500 |
| `--training.tensor_parallel_degree` | Tensor parallelism degree | 4 |
| `--training.data_parallel_replicate_degree` | Data parallelism degree | 2 |
| `--experimental.context_parallel_degree` | Context parallelism degree | 1 |
| `--training.attn_impl` | Attention implementation (`flash_attention_2`, `sdpa`) | `flash_attention_2` |
| `--optimizer.lr` | Learning rate | 2e-5 |
| `--checkpoint.interval` | Checkpoint save interval (steps) | 100 |
| `--training.rope_type` | RoPE scaling type (e.g., `yarn`, `longrope`) | None |
| `--training.rope_factor` | RoPE scaling factor | 1.0 |
