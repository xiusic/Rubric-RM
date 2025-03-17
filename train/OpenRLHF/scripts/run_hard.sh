set -x

export CUDA_VISIBLE_DEVICES=4,5,6

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/shared/nas2/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /shared/nas2/xiusic/gaotang/ckpt/curriculum_chat/llama3-8b-rlhf-all-easy-new-OpenRLHF \
   --save_path /shared/nas2/xiusic/gaotang/ckpt/curriculum_chat/llama3-8b-rlhf-all-hard-new-OpenRLHF \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 5 \
   --max_epochs 1 \
   --prompt_max_len 6850 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.001 \
   --prompt_data /shared/nas2/xiusic/gaotang/data/merged_data_curriculum_easy \
   --input_key context_messages \
   --label_key winner \
   --apply_chat_template \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 3 \
   --ckpt_path /shared/nas2/xiusic/gaotang/ckpt/curriculum_chat/llama3-8b-rlhf-all-hard-new-OpenRLHF \
   --use_wandb 7595e33990e2af809f914f13cefa202fc8fba1ee \
   --wandb_project Rubric-RM-New \
   --wandb_run_name Label_Swap_Llamma3-rlhf-all-merged_data-hard-curriculum \
   --remote_rm_url /shared/nas2/xiusic/OpenRLHF/custom_reward_function.py \
   --verbose_training \
   --verbose_directory /shared/nas2/xiusic/gaotang/logs/curriculum_hard/llama3-8b-rlhf-all-hard-new-OpenRLHF \
   # --normalize_reward \