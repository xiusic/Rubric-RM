
model="wzq016/qwen25-code-math-evidence-rubric-4k2k-separate"
device=7
model_save_name="qwen7b-evidence-rubric-4k2k-separate"
mode="rubric_evidence"
vllm_gpu_util=0.45

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_1.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_2.json --vllm_gpu_util $vllm_gpu_util
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_3.json --vllm_gpu_util $vllm_gpu_util

python scripts/process_final_result.py --model_save_name $model_save_name --model $model