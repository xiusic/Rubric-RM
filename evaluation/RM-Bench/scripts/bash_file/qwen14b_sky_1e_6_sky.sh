
model="/shared/nas2/xiusic/gaotang/skylab-v02/ckpt/qwen2_5-14b-filtered-sky-2_5kcode-18kmath-grpo-flexible-reward-kl-1e-3-lr-1e-6"
device=7
model_save_name="qwen14b_1e-6_sky"
mode="rubric_rl_new"

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_1.json
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_2.json
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --$mode --model $model --datapath data/total_dataset_3.json

python scripts/process_final_result.py --model_save_name $model_save_name --model $model