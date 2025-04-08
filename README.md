# Rubric-RM

Update: [03/15]

This repository contains the training code for the project "Rubric-RM". 

## Installation 

The main dependencies are vllm and flash-attention and OpenRLHF (from source). PyTorch should be automatically downloaded when installing vllm. The following combination of CUDA/Gcc/vllm is tested OK on my end: Cuda-12.4/Gcc-11.3/vllm-0.7.2. You could adjust this according to your environments

```
conda create -n RubricRM python=3.11 -y 
conda activate RubricRM 
pip install uv && uv pip install --upgrade pip
uv pip install vllm==0.7.2
cd Rubric-RM/train/OpenRLHF
uv pip install -e .
```

If the last line doesn't execute correctly, simply do `pip install -e .`.

## Verl environment Installation
Install verl

```
git clone https://github.com/volcengine/verl
cd verl
git checkout e49fb572bf85a8f0ef7124c898f509bd6d9832a1
pip install -e .
```

Install vLLM

```
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout ed6e9075d31e32c8548b480a47d1ffb77da1f54c
git cherry-pick caac5c2e597b1780c3df54a537c34e6061c32cff
export VLLM_COMMIT=ed6e9075d31e32c8548b480a47d1ffb77da1f54c
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

Please install exactly like this, even if you install vllm before, or installed by verl, any other vllm version will fail. In addition, install flash-attention as follows:

```
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install torchdata
```


## How to run training 

First login in your huggingface and wandb (you could use my wandb) as follows:

```
huggingface-cli login
wandb login
```

Second, modify the two variables inside the script: `GLOBAL_WORKINGDIR` and `META_PREFIX` inside the `run_scripts/run_bowen_unhacked_flexible_kl_free.sh` 
and `run_scripts/run_bowen_unhacked_flexible_kl_1e3.sh`. Explanations of the two variables are included inside the bash script. There is no need to modify anything else.


Finally, follow the procedures below:

```
# (to support backend running)
tmux
cd train/OpenRLHF
conda activate RubricRM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head --num-gpus 8 
bash run_scripts/run_bowen_unhacked_flexible_kl_free.sh; bash run_scripts/run_bowen_unhacked_flexible_kl_1e3.sh
```


## Helpful tips

In case you encounter runtime errors (mismatch between ray LD_LIBRARY_PATH and your system's), one possible solution is that 

```
module load cuda-toolkit/12.4
conda activate RubricRM
export LD_LIBRARY_PATH="/software/cuda-12.4/lib64:$LD_LIBRARY_PATH"

# Then start Ray and submit the job within that same shell:
ray start --head
ray job submit --working-dir . -- python your_script.py
```

Update [03/20]

Existing datasets (in huggingface):

- `gaotang/sky_v02_processed_llamma3` (for sky-v02-llama-3.1-8B-rl)
- `gaotang/sky_v02_processed_llamma3_sft` (for sky-v02-llama-3.1-8B-sft)
- `gaotang/sky_v02_processed_qwen` (for sky-v02-qwen-rl)

Note: the first two differ in system prompt (no need to do COT rubrics for sft model), and the final one substitutes 
all the `<im_start>` and `<im_end>` tokens inside the dataset with `<|begin_of_text|>` and `<|end_of_text|>` respectively.

Reward functions that we use are implemented by the following two files

- `train/OpenRLHF/reward_function_flexible.py`
- `train/OpenRLHF/custom_reward_function.py`

Note that the first one is preferred after experiments.

Model checkpoints:

- `meta-llama/Llama-3.1-8B-Instruct` (original; untrained)
- `gaotang/llama3-skywork-v02-sft` (sft)

[Update 03/21]

The latest data:

- `gaotang/sky_v02_filtered` (filtered skywork data, containing 55k of the original 77k data)