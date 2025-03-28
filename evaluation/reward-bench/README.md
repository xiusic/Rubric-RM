This code implements the evaluation pipeline of `reward_bench`. A typical typical prompt looks like the following 

```
# SFT:
CUDA_VISIBLE_DEVICES=1 python scripts/run_generative.py --model $model_path --model_save_name $save_name --sft_new --vllm_gpu_util 0.9

# RL:
CUDA_VISIBLE_DEVICES=1 python scripts/run_generative.py --model $model_path --model_save_name $save_name --rubric_rl_new --vllm_gpu_util 0.9
```

Examples of current sft:

```
CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/llama3-8b-sft-adjusted --model_save_name llama3-8b-sft-math-code-sky --sft_new --vllm_gpu_util 0.9 --stop_llamma3

CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-7b-sft-adjusted --model_save_name qwen-7b-sft-math-code-sky --sft_new --vllm_gpu_util 0.9

CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-14b-sft-adjusted --model_save_name qwen-14b-sft-math-code-sky --sft_new --vllm_gpu_util 0.9
```