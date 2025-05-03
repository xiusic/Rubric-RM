# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --model gpt-3.5-turbo
# python scripts/run_generative.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch 

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from fastchat.conversation import get_conv_template
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import openai 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import convert_robust_dataset_to_preference_dataset_list, load_eval_dataset, compute_accuracy, compute_accuracy_gen
import gc



# from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_answers,
    process_judgement,
    run_judge_pair,
)
from rewardbench.utils import calculate_scores_per_section

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",  # allow list of models (ensemble)
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    # TODO: add dataset PKU-safest
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    # parser.add_argument("--vllm_max_seq_length", type=int, default=None, help="max sequence length for vllm")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    parser.add_argument(
        "--rubric_rl", action="store_true", default=False, help="use rubric_rl chat template for rubric_rl models"
    )
    parser.add_argument(
        '--rubric_evidence', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        "--rubric_rl_new", action="store_true", default=False, help="use rubric_rl chat template for rubric_rl models"
    )
    parser.add_argument(
        '--rubric', action='store_true', default=False, help='use rubric chat template for models that use a rubric'
    )
    parser.add_argument(
        '--sft', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--sft_new', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--original', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--sft_new_user', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--icl', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--icl_openai', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--guideline', action='store_true', default=False, help='use sft chat template for models that use a rubric'
    )
    parser.add_argument(
        '--rubric_rl_rubric', action='store_true', default=False, help='use rubric_rl chat template for models that use a rubric'
    )
    parser.add_argument(
        '--rubric_evidence_classify', action='store_true', default=False, help='use rubric_rl chat template for models that use a rubric'
    )
    parser.add_argument(
        '--rubric_evidence_classify_weight', action='store_true', default=False, help='use rubric_rl chat template for models that use a rubric'
    )
    parser.add_argument(
        '--ablation_no_rubric', action='store_true', default=False, help='use rubric_rl chat template for models that use a rubric'
    )
    parser.add_argument(
        '--reasoning', action='store_true', default=False, help='use rubric_rl chat template for models that use a rubric'
    )
    parser.add_argument(
        '--stop_llamma3', action='store_true', default=False, help='use rubric_rl chat template for models that use a rubric'
    )
    parser.add_argument(
        "--model_save_name", default="default_save", type=str
    )
    parser.add_argument(
        '--max_tokens', type=int, default=2048, help='max tokens for generation'
    )
    parser.add_argument("--datapath", type=str, default="data/reward-bench", help="path to data")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # import json 
    # with open(args.datapath, 'r') as f:
    #     robust_dataset = json.load(f)
    # exit()
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        # assert that is odd and > 1
        assert len(args.model) % 2 == 1

    # define variable if is API or local
    if args.force_local:
        is_api_models = False
    else:
        is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST

    # if model isn't API, load via vllm
    if not is_api_models:
        # if multi gpu, set multiproc method to spawn
        if args.num_gpus > 1:
            # Set the environment variable
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
        #     from transformers import AutoModelForCausalLM
        #     model = AutoModelForCausalLM.from_pretrained(
        #         args.model,
        #         attn_implementation
        #     )
        #     model = model.cuda()
        # else:
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            # max_seq_length=args.vllm_max_seq_length,
            # max_model_len=10000,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model or args.stop_llamma3:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None

    # handle off-case models
    # use different prompt for prometheus/gemini models
    if "prometheus" in args.model:
        model_modifier = "prometheus"
    elif "Con-J" in args.model:
        model_modifier = "Con-J"
    elif "OffsetBias" in args.model:
        model_modifier = "offsetbias"
    elif "Atla" in args.model:
        logger.info("Using ATLA model")
        model_modifier = "Atla"
    elif "gemini" in args.model:
        model_modifier = "gemini"
    else:
        model_modifier = None

    if args.rubric_rl:
        model_modifier = 'rubric_rl'
    if args.rubric:
        model_modifier = 'rubric'
    if args.rubric_rl_rubric:
        model_modifier = 'rubric_rl_rubric'
    if args.rubric_evidence_classify:
        model_modifier = 'rubric_evidence_classify'
    if args.rubric_evidence_classify_weight:
        model_modifier = 'rubric_evidence_classify_weight'
    if args.ablation_no_rubric:
        model_modifier = 'ablation_no_rubric'
    if args.reasoning:
        model_modifier = 'reasoning'
    if args.rubric_evidence:
        model_modifier = "rubric_evidence"
    if args.sft:
        model_modifier = 'sft'
    if args.rubric_rl_new:
        model_modifier = 'rubric_rl_new'
    if args.sft_new:
        model_modifier = "sft_new"
    if args.original:
        model_modifier = "original"
    if args.sft_new_user:
        model_modifier = "sft_new_user"
    if args.icl:
        model_modifier = "icl"
    if args.icl_openai:
        model_modifier = "icl_openai"
    if args.guideline:
        model_modifier = "guideline"

    ############################
    # Load dataset
    ############################

    raw_dataset_list = convert_robust_dataset_to_preference_dataset_list(args.datapath) 
    META_RESULTS_LIST = []
    META_OUTPUT_LIST = []
    META_SHUFFLED = []
    score_chosen, score_rejected = [], []

    # print(raw_dataset_list)
    # exit()
    if args.dataset is None:
        logger.info("*** Load dataset ***")

        for dataset_idx, raw_dataset in enumerate(raw_dataset_list):
        
            # clear cuda memory cache
            dataset = None
            dataloader = None
            torch.cuda.synchronize()
            del dataset
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # prin the gpu memory usage
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
            
            # for device in range(torch.cuda.device_count()):
            #     cuda.select_device(device)  # Select the GPU device
            #     cuda.close()  # Clear the memory
            #     cuda.select_device(device)  # Reinitialize the GPU device if necessary
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")


            dataset, subsets = load_eval_dataset(
                raw_dataset,
                core_set=not args.pref_sets,
                conv=get_conv_template("raw"),
                custom_dialogue_formatting=True,
                tokenizer=tokenizer,
                logger=logger,
                keep_columns=["text_chosen", "text_rejected", "id"],
            )
            # copy id for saving, then remove
            ids = dataset["id"]
            dataset = dataset.remove_columns("id")

            # debug: use only 10 examples
            if args.debug:
                dataset = dataset.select(range(10))
                subsets = subsets[:10]
                ids = ids[:10]
            
            # dataset, subsets = load_eval_dataset(
            #     core_set=not args.pref_sets,
            #     conv=get_conv_template("raw"),  # not used in this script (handled later)
            #     custom_dialogue_formatting=True,  # handle formatting later
            #     tokenizer=None,
            #     logger=logger,
            #     keep_columns=["text_chosen", "text_rejected", "id"],
            #     max_turns=4,
            # )
            # dataset format:
            # Dataset({
            #     features: ['id', 'text_chosen', 'text_rejected'],
            #     num_rows: 2985
            # })        

            if is_api_models:
            ############################
            # Run inference via API
            ############################
                pass 
            else:
                
                ############################
                # Run model weights with vllm
                ############################


                if model_modifier == "icl":
                    retrieval_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
                    retrieval_index = faiss.read_index("/shared/nas2/xiusic/Rubric-RM/evaluation/notebook/corpus_faiss.index")

                    with open("/shared/nas2/xiusic/Rubric-RM/evaluation/notebook/corpus_id_to_content.json") as f:
                        id_to_content = json.load(f)
                    meta_dir=None
                elif model_modifier == "icl_openai":

                    def get_openai_key(dir="/shared/nas2/xiusic/openai_key.txt"):
                        with open(dir, 'r') as f:
                            key = f.read()
                        return key 

                    key = get_openai_key()
                    openai.api_key = key 
                    meta_dir = Path("/shared/nas2/xiusic/Rubric-RM/RAG/sky_filtered_code_2_5k_math_18k")  
                    norm = "normalized"
                    retrieval_model = "openai"
                    retrieval_index = faiss.read_index(f"{meta_dir}/{norm}_corpus.index")
                    with open(meta_dir / f"{norm}_id_to_item.json") as f:
                        id_to_content = json.load(f)
                else:
                    retrieval_model=None
                    retrieval_index=None
                    id_to_content=None
                    meta_dir=None

                if model_modifier == "guideline":
                    file_path = "/shared/nas2/xiusic/Rubric-RM/document_guideline/model_spec_chat.txt"

                    with open(file_path, "r", encoding="utf-8") as file:
                        # Read the entire file into a single string
                        guideline_document = file.read()
                else:
                    guideline_document = None 

                def format_judgements(batch, optional_chat_template=None):
                    # TODO expand this to include fastchat chat templates if needed
                    mult_turn = True if len(batch["text_chosen"]) > 2 else False

                    if mult_turn:
                        print("Multi turn!")
                        exit()
                    prompt = batch["text_chosen"][0]["content"]
                    answer_a = batch["text_chosen"]
                    answer_b = batch["text_rejected"]

                    # shuffle a and b randomly for position bias
                    is_shuffled = np.random.rand() > 0.5
                    if is_shuffled:
                        answer_a, answer_b = answer_b, answer_a

                    system_prompt, user_prompt = format_judge_answers(
                        prompt, answer_a, answer_b, multi_turn=mult_turn, model_modifier=model_modifier,
                        retrieval_model=retrieval_model,
                        retrieval_index=retrieval_index,
                        id_to_content=id_to_content,
                        meta_dir=meta_dir,
                        top_k=3,
                        guideline_document=guideline_document
                    )

                    # if np.random.rand() < 0.01:
                    #     print("system_prompt:", system_prompt)
                    #     print("user_prompt:", user_prompt)

                    if optional_chat_template is not None:
                        raise NotImplementedError("Chat templates not implemented yet")
                        optional_chat_template.set_system_message(system_prompt)
                        optional_chat_template.messages = []
                        optional_chat_template.append_message(optional_chat_template.roles[0], user_prompt)
                        optional_chat_template.append_message(optional_chat_template.roles[1], None)
                        prompt = optional_chat_template.get_prompt()
                    else:
                        if args.reasoning:
                            messages = [
                                {'role': "user", 'content':user_prompt},
                            ]
                        else:
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_prompt,
                                },
                                {"role": "user", "content": user_prompt},
                            ]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        # chat template already include special tokens
                        # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
                        # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
                        # so we need to tokenize without adding special tokens
                        tokenized_prompt = tokenizer(prompt, add_special_tokens=False, return_length=True)
                        prompt_ids = tokenized_prompt["input_ids"]

                    batch["text"] = prompt
                    batch["is_shuffled"] = is_shuffled
                    batch["prompt_ids"] = prompt_ids
                    return batch

                # format the dataset for the model, with optional fastchat templating
                if args.chat_template is not None:
                    chat_template = get_conv_template(args.chat_template)
                else:
                    chat_template = None
                
                # print("Before: ", dataset[0])
                dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})
                # print("#" * 100)
                # print("After: ", tokenizer.batch_decode(dataset_prompts[0]['prompt_ids']))
                # exit()


                # collect texts of dataset in list
                prompts = dataset_prompts["text"]
                prompt_ids = dataset_prompts["prompt_ids"]
                is_shuffled = dataset_prompts["is_shuffled"]

                # generate
                logger.info("*** Run inference ***")

                if model_modifier == "Atla":
                    logger.info("Using Atla model for inference")
                    outputs = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
                else:
                    # if args.model == 'meta-llama/Llama-3.1-8B-Instruct':

                    #     outputs = []
                    #     for prompt in tqdm(prompts, total=len(prompts)):
                    #         input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids
                    #         out = model.generate(input_ids.cuda(), max_new_tokens=50)
                    #         import ipdb; ipdb.set_trace()
                    #         outputs.append(tokenizer.decode(out[0][input_ids.shape[1]:]))

                    #     with open("./outputs.json", "w") as file:
                    #         json.dump(outputs, file)

                    # else:
                    sampling_params = SamplingParams(
                        n=1,
                        temperature=0,
                        top_p=1,
                        max_tokens=args.max_tokens,
                        stop_token_ids=stop_token_ids,
                    )
                    outputs = model.generate(prompts, sampling_params=sampling_params)

                    # print(output)
                    # print("model_modifer: ", model_modifier)
                    # exit()
                    logger.info("*** Inference done ***")

                    if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
                        answers = outputs
                    else:
                        answers = [o.outputs[0].text for o in outputs]
                    # print(answers)
                    winners = [process_judgement(a, model_modifier) for a in answers]
                    # print("winners: ", winners) 

                    ds_string = '_pku_safe' if args.dataset == 'PKU-Alignment/PKU-SafeRLHF' else ''

                    import json
                    if args.rubric_rl:
                        with open(f"./output/answers{ds_string}_rubric_rl_{args.model_save_name}_{dataset_idx}.json", "w") as file:
                            json.dump(answers, file, indent=4)
                    elif args.rubric:
                        with open(f"./output/answers{ds_string}_rubric.json", "w") as file:
                            json.dump(answers, file)
                    elif args.rubric_rl_rubric:
                        with open(f"./output/answers{ds_string}_rubric_rl_rubric.json", "w") as file:
                            json.dump(answers, file)
                    elif args.rubric_evidence_classify:
                        with open(f"./output/answers{ds_string}_rubric_evidence_classify.json", "w") as file:
                            json.dump(answers, file)
                    elif args.rubric_evidence_classify_weight:
                        with open(f"./output/answers{ds_string}_rubric_evidence_classify_weight.json", "w") as file:
                            json.dump(answers, file)
                    elif args.ablation_no_rubric:
                        with open(f"./output/answers{ds_string}_ablation_no_rubric.json", "w") as file:
                            json.dump(answers, file)
                    elif args.reasoning:
                        with open(f"./output/answers{ds_string}_reasoning.json", "w") as file:
                            json.dump(answers, file)
                    elif args.original:
                        with open(f"./output/answers{ds_string}_original.json", "w") as file:
                            json.dump(answers, file)
                    elif args.rubric_evidence:
                        with open(f"./output/answers{ds_string}_rubric_evidence.json", "w") as file:
                            json.dump(answers, file)
                    elif args.sft_new:
                        with open(f"./output/answers{ds_string}_sft_new.json", "w") as file:
                            json.dump(answers, file, indent=4)
                    else:
                        with open(f"./output/answers{ds_string}.json", "w") as file:
                            json.dump(answers, file)

                    def process_shuffled(win, shuffle):
                        if shuffle:
                            winner_text = "B"
                            loser_text = "A"
                        else:
                            winner_text = "A"
                            loser_text = "B"

                        if win == winner_text:
                            return 1
                        else:
                            return 0
                    results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]
                    # score_chosen, score_chosen = results, 
                    # print("winners: ", winners)
                    # print("exit")
                    # print("results: ",results)
                    META_RESULTS_LIST.append(results)
                    META_OUTPUT_LIST.append(answers)
                    META_SHUFFLED.append(is_shuffled)



                    # score_chosen.append([1 if i == 1 else 0 for i in results])
                    # score_rejected.append([1 if i == 0 else 0 for i in results])

                    # print("result: ", results)
                    # print("score chosen: ", score_chosen) 
                    # print("score rejected: ", score_rejected)
    
    # exit()
    ############################
    # Save results
    ############################
    
    import json
    # HACK: load the dataset from the file
    dataset_json:list = json.load(open(args.datapath))
    if args.debug:
        dataset_json = dataset_json[:10]
    
    for idx, unit in enumerate(dataset_json):
        # unit['score_orig'] = score_original[idx]
        unit['result'] = [
            res_list[idx] for res_list in META_RESULTS_LIST
        ]
        unit['output'] = [
            output_list[idx] for output_list in META_OUTPUT_LIST
        ]
        unit['Is_Chosen_Answer_Shuffled_toPositionB'] = [
            shuffle_list[idx] for shuffle_list in META_SHUFFLED
        ]

        # unit['score_chosen'] = [
        #     score_list[idx] for score_list in score_chosen
        # ]
        # unit['score_rejected'] = [
        #     score_list[idx] for score_list in score_rejected
        # ]

        # if all the elemnts in the list are list and all the elements is of length 1
        # if all(isinstance(elem, list) and len(elem) == 1 for elem in unit['score_chosen']):
        #     unit['score_chosen'] = [elem[0] for elem in unit['score_chosen']]
        # if all(isinstance(elem, list) and len(elem) == 1 for elem in unit['score_rejected']):
        #     unit['score_rejected'] = [elem[0] for elem in unit['score_rejected']]
    
    # save to results folder with the name + model name + timestamp
    filename = os.path.basename(args.datapath).replace(".json", "")
    model_name = args.model.split("/")[-1]
    ref_model_name = "REWORD_MODEL"
    # make a dir at results with official model name
    output_dir = f"results/Gen_RMs/{args.model_save_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from datetime import datetime
    # output_path = os.path.join(output_dir, f"{filename}_{model_name}_{ref_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path = os.path.join(output_dir, f"{filename}_{model_name}_{ref_model_name}.json")
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
        # acc_dict = compute_accuracy_gen(dataset_json)
    # print(f"The accuracy of model {model_name}\n in the dataset {filename} is:\n {acc_dict}")

    # def get_basic_acc()
    right, total = 0, 0 
    for item in dataset_json:
        total += len(item['result']) 
        right += sum(item['result'])

    print(f"Finished. The ordinary accuracy is: {right / total}")

if __name__ == "__main__":
    main()
