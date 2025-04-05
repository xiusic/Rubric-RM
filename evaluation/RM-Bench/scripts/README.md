```
CUDA_VISIBLE_DEVICES=4 python scripts/run_generative.py --trust_remote_code --stop_llamma3 --model_save_name sft --sft --model /shared/nas2/xiusic/gaotang/skylab-v02-baseline/ckpt/llama3-8b-sft --datapath data/total_dataset1.json

CUDA_VISIBLE_DEVICES=7 python scripts/run_generative.py --trust_remote_code --stop_llamma3 --model_save_name new_code --rubric_rl_new --model wzq016/llama3-skywork-rlrm-filtered-code-grpo-kl --datapath data/total_dataset_1.json

CUDA_VISIBLE_DEVICES=7 python scripts/run_generative.py --trust_remote_code --stop_llamma3 --model_save_name new_code --rubric_rl_new --model wzq016/llama3-skywork-rlrm-filtered-code-grpo-kl --datapath /home/xiusic/RM-Bench/data/total_dataset_2.json

CUDA_VISIBLE_DEVICES=7 python scripts/run_generative.py --trust_remote_code --stop_llamma3 --model_save_name new_code --rubric_rl_new --model wzq016/llama3-skywork-rlrm-filtered-code-grpo-kl --datapath data/total_dataset_3.json
```