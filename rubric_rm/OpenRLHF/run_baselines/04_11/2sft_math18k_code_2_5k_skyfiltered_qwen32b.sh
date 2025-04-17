

deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft \
   --save_path /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-32b-sft-sky_filtered_code_8k_math_10k \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain Qwen/Qwen2.5-32B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset gaotang/filtered_sky_code_8k_math_10k_rubric_sft \
   --apply_chat_template \
   --input_key context_messages \
   --output_key winner \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --adam_offload \
   --use_wandb 7595e33990e2af809f914f13cefa202fc8fba1ee \
   --wandb_project Rubric-RM-baseline \
   --wandb_run_name Qwen-32b-sft-lr5e-7sky_filtered_code_8k_math_10k \