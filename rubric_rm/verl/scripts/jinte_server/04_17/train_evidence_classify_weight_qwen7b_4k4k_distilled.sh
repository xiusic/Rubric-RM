export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="INFO"
N_GPU=8

# Model Setting
MODEL_PATH=gaotang/qwen_7b_sky_filtered_code8k_math_10k_distilled_OpenAI

# Training Setting
LR=1.0e-6
GPU_MEM_UTILIZATION=0.5 # Lower this if you met OOM problem
TOTAL_EPISODES=1
SAVE_EVERY_STEP=100
TEST_EVERY_STEP=100000
TRAIN_BS=1024           # Rollout batchsize. Could be arbitrary large, but must be divided by N_GPU.
PPO_MINI_BS=128         # Train batch size. Could be arbitrary large, must be the divisor of TRAIN_BS and be divided by N_GPU. Setting this equal to TRAIN_BS means strictly on-policy.
MAX_PROMPT_LENGTH=4096  # Lower this if you met OOM problem.
MAX_RESPONSE_LENGTH=4096 # Lower this if you met OOM problem
TRAIN_PER_GPU=4         # REAL train batch size per gpu. Lower this if you met OOM problem. Must be a divisor of PPO_MINI_BS.
FORWARD_PER_GPU=4       # Batch size to get logprob. Lower this if you met OOM problem. Must be a divisor of TRAIN_BS.

# Logging Setting
PROJECT_NAME=rubric_rm
EXPERIMENT_NAME=rubric_rm_qwen2.5_7B_LR${LR}_filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_4k4k_distilled_OpenAI
SAVE_NAME=qwen2.5_7B_LR${LR}_filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_4k4k_distilled_OpenAI
SAVE_META_DIR="Your"


# Reward Setting
REWARD_PATH=./rubric_rm/verl/utils/reward_score/lm_as_judge_evidence_rubric_classify_separate_reward.py
REWARD_FUNC_NAME=lm_as_judge_match

# Task
TRAIN_TASK="gaotang/filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_rest_0417"
EVAL_TASK="gaotang/filtered_sky_code_8k_math_10k_rubric_evidence_classify_weight_rest_0417"

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
    trainer.test_freq=-1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    trainer.default_local_dir=${SAVE_META_DIR}/${SAVE_NAME}

ray stop


# A=$(ls checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_* | sort -t_ -k3 -n | tail -n1 | sed 's/:$//')
# python rubric_rm/verl/scripts/converter.py \
#     --local_dir "$A/actor" \
#     --hf_upload_path "wzq016/${EXPERIMENT_NAME}"