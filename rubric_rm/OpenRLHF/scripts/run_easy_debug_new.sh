set -x

export CUDA_VISIBLE_DEVICES=0,1,2

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/shared/nas2/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --save_path /shared/nas2/xiusic/gaotang/ckpt/debug/llamma-8B-chat-easy-curriculum \
   --micro_train_batch_size 1 \
   --train_batch_size 4 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt 2 \
   --max_epochs 1 \
   --prompt_max_len 6850 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0 \
   --prompt_data /shared/nas2/xiusic/gaotang/data/merged_data_curriculum_easy \
   --input_key context_messages \
   --label_key winner \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 15 \
   --ckpt_path /shared/nas2/xiusic/gaotang/ckpt/debug/llamma-8B-chat-easy-curriculum \
   --use_wandb 7595e33990e2af809f914f13cefa202fc8fba1ee \
   --wandb_project Debug-Rubric-RM \
   --wandb_run_name debug-llamma-8b \
   --remote_rm_url /shared/nas2/xiusic/OpenRLHF/custom_reward_function.py \
   --verbose_training \
   --verbose_directory /shared/nas2/xiusic/gaotang/logs/debug/llamma-8B-chat-easy-curriculum \