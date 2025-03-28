```
CUDA_VISIBLE_DEVICES=1 python /home/xiusic/reward-bench/scripts/run_generative.py --model wzq016/llama3-skywork-rlrm-filtered-grpo-kl --stop_llamma3 --model_save_name filtered_llama3 --rubric_rl_new --vllm_gpu_util 0.75

CUDA_VISIBLE_DEVICES=7 python /home/xiusic/reward-bench/scripts/run_generative.py --model wzq016/llama3-skywork-rlrm-filtered-code-grpo-kl --stop_llamma3 --model_save_name filtered_llama3_code --rubric_rl_new

CUDA_VISIBLE_DEVICES=6 python /home/xiusic/reward-bench/scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-14b-sft --model_save_name filtered_qwen_14b_code_math-sft --sft_new --vllm_gpu_util 0.9
```