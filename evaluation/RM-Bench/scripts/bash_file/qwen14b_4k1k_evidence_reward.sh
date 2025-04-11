
model="wzq016/rubric_rm_qwen2.5_14B_LR1.0e-6_sky_filtered_code_2_5k_math_18k_evidence_rubric"
device=4
model_save_name="qwen14b-evidence-rubric-4k1k"
mode="rubric_evidence"
vllm_gpu_util=0.9

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_1.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_2.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_3.json --vllm_gpu_util $vllm_gpu_util

python scripts/process_final_result.py --model_save_name $model_save_name --model $model