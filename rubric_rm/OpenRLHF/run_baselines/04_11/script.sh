bash /shared/nas2/xiusic/Rubric-RM/rubric_rm/OpenRLHF/run_baselines/04_11/2sft_math18k_code_2_5k_skyfiltered_qwen7b.sh
bash /shared/nas2/xiusic/Rubric-RM/rubric_rm/OpenRLHF/run_baselines/04_11/2sft_math18k_code_2_5k_skyfiltered_qwen14b.sh 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /shared/nas2/xiusic/train.py