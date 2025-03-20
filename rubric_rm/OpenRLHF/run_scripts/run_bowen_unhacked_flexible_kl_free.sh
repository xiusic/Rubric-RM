#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## Define the right variables

# GLOBAL_WORKINGDIR is simply the dir of OpenRLHF, please use the absolute path in your system, like the example provided
GLOBAL_WORKINGDIR="Your global working dir to OpenRLHF. e.g. /shared/nas2/xiusic/Rubric-RM/train/OpenRLHF"

# META_PREFIX defines the storage path, i.e. where you store the logs and model checkpoints
META_PREFIX="Your MetaStorage path. e.g. /shared/nas2/xiusic/gaotang"

# The model save path is handled for you, format: meta-storage-path / dataset-name / ckpt / model-setting
USE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAVE_MODEL_NAME="llamma3-8b-grpo-flexible-reward-no-kl"
SAVE_MODEL_PREFIX="$META_PREFIX/skylab-v02/ckpt"
SAVE_PATH="$SAVE_MODEL_PREFIX/$SAVE_MODEL_NAME"
VERBOSE_DIR="$META_PREFIX/skylab-v02/logs/$SAVE_MODEL_NAME"
DATASET_PATH="gaotang/sky_v02_processed_llamma3"

# Submit the Ray job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json "{\"working_dir\":\"$GLOBAL_WORKINGDIR\"}" \
   -- \
   python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 0 \
   --ref_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --pretrain "$USE_MODEL" \
   --save_path "$SAVE_PATH" \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 7 \
   --max_epochs 1 \
   --prompt_max_len 6850 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0 \
   --prompt_data $DATASET_PATH \
   --input_key context_messages \
   --label_key winner \
   --apply_chat_template \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 5 \
   --save_hf_ckpt \
   --load_checkpoint \
   --ckpt_path "$SAVE_PATH" \
   --use_wandb 7595e33990e2af809f914f13cefa202fc8fba1ee \
   --wandb_project Rubric-RM-new-Bowen \
   --wandb_run_name Label_Swap_Llamma3-rlhf-all-skylab-v02-grpo-unhacked-flexible-nokl \
   --remote_rm_url "$GLOBAL_WORKINGDIR/reward_function_flexible.py" \
   --verbose_training \
   --verbose_directory "$VERBOSE_DIR" \