
model="/shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/qwen2.5_14B_LR1.0e-6_evidence_rubric_4k2k_separate_reward_function/global_step_73/actor/huggingface"
device=2
model_save_name="qwen14b-separate-reward-4k2k"
mode="rubric_evidence"
vllm_gpu_util=0.9

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_1.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_2.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_3.json --vllm_gpu_util $vllm_gpu_util

python scripts/process_final_result.py --model_save_name $model_save_name --model $model