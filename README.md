# $\infty$-THOR: Beyond Needle(s) in the Embodied Haystack

We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI.

$\infty$-THOR provides:

(1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories;

(2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents’ long-context reasoning ability; and 

(3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences.

To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.


<p align="center">
| <a href="https://arxiv.org/pdf/2505.16928"><b>Paper</b></a> | <a href="https://huggingface.co/datasets/PEARLS-Lab/infini-thor"> <b>Dataset</b> </a> | <a href="https://x.com/rajammanabrolu/status/1925945858664604025"> <b>Tweet</b> </a> |
</p>


## Setup

We provide a Dockerfile for setting up the environment. To build the image:
```
docker build -t infini-thor -f Dockerfile .
```
or pull iamge from the hub
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

## Static Evaluation: Needle(s) in the Embodied Haystack (NiEH)

### Download and unzip QA Data
Download the NiEH set from huggingface dataset hub (you may need to set HF_TOKEN or login with `huggingface-cli login`)
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download PEARLS-Lab/infini-thor --repo-type dataset --local-dir /path/to/directory
```
or
```
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='PEARLS-Lab/infini-thor', repo_type='dataset', local_dir='/path/to/directory')" 

```

Download images and metadata and uncompress
```
wget https://huggingface.co/PEARLS-Lab/infini-thor/resolve/main/dataset/testset.tar
tar xvf testset.tar
```

### Data Format

**NiEH Data File (CSV)**:
- `qa_set_nieh_single_clue.csv`: Single-evidence QA set (Needle in the Emboided Haystack task)
- `qa_set_nsieh_multi_clue.csv`: Multi-evidnece QA set (Needle**s** int the Embodied Haystack task)

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

### Run evaluation

```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf
```

Running with a context extension method
e.g.,
```bash
python run_eval_QA_NiEH.py \
    --qa_file_path path/to/qa_data.csv \
    --metadata_dir path/to/metadata \
    --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --ctx_extension yarn \
    --ctx_extension_factor 4.0
```

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
Use the script below to start a virtual X server (e.g., with Xvfb) on display :0:
```
# use tmux or run in background
conda activate ai2thor_env
python env_utils/startx.py 0
```

**Running AI2THOR service**

We use a microservices to solve the version compatibility issue between AI2THOR (python 3.6) and PyTorch (python 3.10 or later) environments.
`ai2thor_service.py` runs the AI2THOR simulator, provides a REST API for environment interactions and handles all scene management and agent actions.
This works over the Flask and let us communicate between the simulator and agent over TCP.

```
# use tmux or run in background
python env_utils/ai2thor_service.py
```

Running the evaluation:

```
export MODEL_LABEL=llava_onevison_7b_32k
python run_interactive_eval.py \
  --checkpoint checkpoints/$MODEL_LABEL \
  --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --flash_attn
```

## Fine-tuning VLA
(coming soon!)