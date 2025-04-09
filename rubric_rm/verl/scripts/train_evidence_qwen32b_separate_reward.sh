#!/bin/bash
#SBATCH --job-name=rl                                # Job name
#SBATCH --nodes=2                                    # Number of nodes
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
export VERL_PPO_LOGGING_LEVEL="INFO"
N_GPU=8

# Model Setting
MODEL_PATH=/mnt/home/ziqi/hf_model/Qwen2.5-32B-Instruct

# Training Setting
LR=1.0e-6
GPU_MEM_UTILIZATION=0.5 # Lower this if you met OOM problem
TOTAL_EPISODES=1
SAVE_EVERY_STEP=100
TEST_EVERY_STEP=100000
TRAIN_BS=1024           # Rollout batchsize. Could be arbitrary large, but must be divided by N_GPU.
PPO_MINI_BS=128         # Train batch size. Could be arbitrary large, must be the divisor of TRAIN_BS and be divided by N_GPU. Setting this equal to TRAIN_BS means strictly on-policy.
MAX_PROMPT_LENGTH=4096  # Lower this if you met OOM problem.
MAX_RESPONSE_LENGTH=2048 # Lower this if you met OOM problem
TRAIN_PER_GPU=1         # REAL train batch size per gpu. Lower this if you met OOM problem. Must be a divisor of PPO_MINI_BS.
FORWARD_PER_GPU=1       # Batch size to get logprob. Lower this if you met OOM problem. Must be a divisor of TRAIN_BS.

# Logging Setting
PROJECT_NAME=rubric_rm
EXPERIMENT_NAME=rubric_rm_qwen2.5_32B_LR${LR}_sky_filtered_code_2_5k_math_18k_evidence_rubric_4k2k_separate_reward

# Reward Setting
REWARD_PATH=./rubric_rm/verl/utils/reward_score/lm_as_judge_evidence_rubric_separate_reward.py
REWARD_FUNC_NAME=lm_as_judge_match

# Task
TRAIN_TASK="gaotang/sky_v02_filtered_2_5kcode_18kmath_evidence_evaluation_justify_rubric"
EVAL_TASK="gaotang/sky_v02_filtered_2_5kcode_18kmath_evidence_evaluation_justify_rubric"

# FIXED SETTING (DO NOT MODIFY IF YOU DO NOT KNOW WHAT IT MEANS)
MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

# Incase the node has ray engine started.
ray stop
sleep 5
ray stop

# --------------------------
# Prepare Ray Cluster
# --------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get the head node IP (strip out IPv6 if present)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi

PORT=6379
export ip_head="${head_node_ip}:${PORT}"
echo "Head Node: $head_node"
echo "Head IP: $head_node_ip"
echo "Ray Head Address: $ip_head"


# Stop Ray if already running, then start fresh
ray stop -f
sleep 3


# --------------------------
# Start Ray Head
# --------------------------
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
              --node-ip-address "$head_node_ip" \
              --port $PORT \
              --num-cpus $SLURM_CPUS_PER_TASK \
              --num-gpus $N_GPU \
              --block &
sleep 10

# --------------------------
# Start Ray Workers
# --------------------------
worker_num=$(( SLURM_JOB_NUM_NODES - 1 ))
for (( i=1; i<=$worker_num; i++ )); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray worker on $node_i..."
  srun --nodes=1 --ntasks=1 -w "$node_i" \
      ray start --address "$ip_head" \
                --num-cpus $SLURM_CPUS_PER_TASK \
                --num-gpus $N_GPU \
                --block &
  sleep 5
done

# Give workers time to connect
sleep 10


# ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU}

# --------------------------
# Run VERL Training on Head
# --------------------------

srun --nodes=1 --ntasks=1 -w "$head_node" \
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
    trainer.n_gpus_per_node=${N_GPU} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \

# Lines 145 - 150 helps with model parallelism https://github.com/volcengine/verl/issues/263

ray stop
