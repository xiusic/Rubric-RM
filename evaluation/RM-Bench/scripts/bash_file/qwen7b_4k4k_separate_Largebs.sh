
model="/shared/nas2/xiusic/gaotang/ckpt/other_run/qwen2.5_7B_LR5.0e-6_evidence_rubric_4k4k_separate_reward_function_largeBz/global_step_36"
device=1
model_save_name="qwen7b-evidence-rubric-4k4k_separate_largebs"
mode="rubric_evidence"
vllm_gpu_util=0.45

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_1.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_2.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_3.json --vllm_gpu_util $vllm_gpu_util

python scripts/process_final_result.py --model_save_name $model_save_name --model $model