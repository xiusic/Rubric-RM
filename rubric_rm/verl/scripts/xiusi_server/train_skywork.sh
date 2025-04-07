set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="INFO"
N_GPU=8

# Model Setting
# MODEL_PATH=/mnt/home/ziqi/checkpoints/Llama-3.1-8B-Instruct
MODEL_PATH=/mnt/home/ziqi/hf_model/Qwen2.5-7B-Instruct


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
EXPERIMENT_NAME=rubric_rm_qwen2.5_LR${LR}_entire_orm_guideline_grpo_kl_8k

# Reward Setting
REWARD_PATH=./rubric_rm/verl/utils/reward_score/lm_as_judge.py
REWARD_FUNC_NAME=lm_as_judge_match

# Task
TRAIN_TASK="gaotang/entire_orm_guideline"
EVAL_TASK="gaotang/entire_orm_guideline"

# Incase the node has ray engine started.
ray stop
sleep 5
ray stop

export CUDA_VISIBLE_DEVICES=4,5,6,7
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
