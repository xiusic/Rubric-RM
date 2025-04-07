
model="wzq016/qwen25-entrie-guideline-8k"
device=3
model_save_name="entire_guideline_76k"
mode="guideline"

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_1.json
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_2.json
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_3.json

python scripts/process_final_result.py --model_save_name $model_save_name --model $model