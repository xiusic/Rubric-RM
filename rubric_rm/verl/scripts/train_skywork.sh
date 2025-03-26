#!/bin/bash
#SBATCH --job-name=rl                                # Job name
#SBATCH --nodes=1                                    # Number of nodes
#SBATCH --ntasks-per-node=1                          # Number of tasks per node
#SBATCH --cpus-per-task=128                          # Number of CPUs per task
#SBATCH --gres=gpu:8                                 # Number of GPUs per node
#SBATCH --gpus-per-task=8                            # Number of GPUs per task
#SBATCH --exclusive                                  # Use the entire node, all CPUs and memory
#SBATCH --time=72:00:00                              # Maximum runtime
#SBATCH --output=./logs/output_%j.log                # Standard output and error log

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="DEBUG"
N_GPU=8

# Model Setting
MODEL_PATH=/mnt/home/ziqi/checkpoints/Llama-3.1-8B-Instruct
# MODEL_PATH=/mnt/home/ziqi/hf_model/Qwen2.5-7B-Instruct
# MODEL_PATH=/mnt/home/ziqi/checkpoints/llama3-skywork-v02-sft


# Training Setting
LR=1.0e-6
GPU_MEM_UTILIZATION=0.5
TOTAL_EPISODES=1
SAVE_EVERY_STEP=100
TEST_EVERY_STEP=100000
TRAIN_BS=1024
PPO_MINI_BS=128

# Logging Setting
PROJECT_NAME=rubric_rm
EXPERIMENT_NAME=rubric_rm_llama3_LR${LR}_new_filtered_code5k_math18k_grpo_kl_formatting_constraint

# Reward Setting
REWARD_PATH=./rubric_rm/verl/utils/reward_score/lm_as_judge.py
REWARD_FUNC_NAME=lm_as_judge_match

# Task
TRAIN_TASK="gaotang/sky_v02_filtered_2_5kcode_18kmath_math_code_sky"
EVAL_TASK="gaotang/sky_v02_filtered_2_5kcode_18kmath_math_code_sky"

# Incase the node has ray engine started.
ray stop
sleep 5
ray stop

ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU}

python3 -m rubric_rm.verl.trainer.main_ppo \
    data.train_files=${TRAIN_TASK} \
    data.val_files=${EVAL_TASK} \
    data.train_batch_size=${TRAIN_BS} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTILIZATION} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.total_epochs=${TOTAL_EPISODES} \
    trainer.save_freq=${SAVE_EVERY_STEP} \
    trainer.test_freq=${TEST_EVERY_STEP} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU}

ray stop
