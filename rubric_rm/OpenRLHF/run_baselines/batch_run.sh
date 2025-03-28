#!/bin/bash

bash /shared/nas2/xiusic/Rubric-RM/rubric_rm/OpenRLHF/run_baselines/2sft_math18k_code_2_5k_skyfiltered_qwen14b.sh

bash /shared/nas2/xiusic/Rubric-RM/rubric_rm/OpenRLHF/run_baselines/2sft_math18k_code_2_5k_skyfiltered_qwen7b.sh

bash /shared/nas2/xiusic/Rubric-RM/rubric_rm/OpenRLHF/run_baselines/2sft_lmath18k_code_2_5k_skyfiltered_llama8b.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /home/xiusic/Rubric-RM/OpenRLHF/train.py