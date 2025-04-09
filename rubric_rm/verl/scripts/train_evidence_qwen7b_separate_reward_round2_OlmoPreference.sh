#!/bin/bash
#SBATCH --job-name=rl                                # Job name
#SBATCH --nodes=1                                    # Number of nodes
#SBATCH --ntasks-per-node=1                          # Number of tasks per node
#SBATCH --cpus-per-task=128                          # Number of CPUs per task
#SBATCH --gres=gpu:8                                 # Number of GPUs per node
#SBATCH --gpus-per-task=8                            # Number of GPUs per task
#SBATCH --exclusive                                  # Use the entire node, all CPUs and memory
#SBATCH --time=144:00:00                              # Maximum runtime
#SBATCH --output=./logs/output_%j.log                # Standard output and error log

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="INFO"
N_GPU=8

# Training Setting
LR=5.0e-7
GPU_MEM_UTILIZATION=0.5 # Lower this if you met OOM problem
TOTAL_EPISODES=1
SAVE_EVERY_STEP=50
TEST_EVERY_STEP=100000
TRAIN_BS=1024           # Rollout batchsize. Could be arbitrary large, but must be divided by N_GPU.
PPO_MINI_BS=128         # Train batch size. Could be arbitrary large, must be the divisor of TRAIN_BS and be divided by N_GPU. Setting this equal to TRAIN_BS means strictly on-policy.
MAX_PROMPT_LENGTH=4096  # Lower this if you met OOM problem.
MAX_RESPONSE_LENGTH=2048 # Lower this if you met OOM problem
TRAIN_PER_GPU=4         # REAL train batch size per gpu. Lower this if you met OOM problem. Must be a divisor of PPO_MINI_BS.
FORWARD_PER_GPU=8       # Batch size to get logprob. Lower this if you met OOM problem. Must be a divisor of TRAIN_BS.

# Logging Setting
PROJECT_NAME=rubric_rm
EXPERIMENT_NAME=rubric_rm_qwen2.5_7B_LR${LR}_sky_filtered_code_2_5k_math_18k_evidence_rubric_4k2k_separate_reward_secondIter_Olmo


# Model Setting
MODEL_PATH=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

# Reward Setting
REWARD_PATH=./rubric_rm/verl/utils/reward_score/lm_as_judge_evidence_rubric_separate_reward.py
REWARD_FUNC_NAME=lm_as_judge_match

# Task
TRAIN_TASK="gaotang/Olmo2_preference_dataset"
EVAL_TASK="gaotang/Olmo2_preference_dataset"

# FIXED SETTING (DO NOT MODIFY IF YOU DO NOT KNOW WHAT IT MEANS)
MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

# Incase the node has ray engine started.
ray stop
sleep 5
ray stop

ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU}

python3 -m rubric_rm.verl.trainer.main_ppo \
    data.train_files=${TRAIN_TASK} \
    data.val_files=${EVAL_TASK} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BS} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TRAIN_PER_GPU} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.total_epochs=${TOTAL_EPISODES} \
    trainer.save_freq=${SAVE_EVERY_STEP} \
    trainer.test_freq=${TEST_EVERY_STEP} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU}

ray stop
