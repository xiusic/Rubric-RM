

deepspeed --include localhost:0,1,2,3 --module openrlhf.cli.train_sft \
   --save_path /shared/nas2/xiusic/gaotang/skylab-v02-baseline/ckpt/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset gaotang/sky_v02_processed_llamma3_sft \
   --apply_chat_template \
   --input_key context_messages \
   --output_key winner \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --adam_offload \
   --use_wandb 7595e33990e2af809f914f13cefa202fc8fba1ee \
   --wandb_project Rubric-RM-baseline \
   --wandb_run_name Llamma3-8b-sft-lr5e-7skywork \