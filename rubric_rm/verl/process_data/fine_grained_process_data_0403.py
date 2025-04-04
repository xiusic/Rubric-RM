import copy, re
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import Counter

import argparse 

def parse_embedded_mpt_blocks(content: str, default_role: str = "assistant"):
    """
    Parse a single message 'content' string that may contain embedded
    <|im_start|>role\n ... <|im_end|> blocks (MPT-style).
    
    Returns a list of { "role": ..., "content": ... }.
    
    - Any text before the first <|im_start|> is assigned to `default_role`.
    - Each <|im_start|>block...<|im_end|> is split out as its own message,
      using the line immediately after <|im_start|> as the role.
    - If a given block has empty content, it's skipped.
    """
    # If no <|im_start|> in the string, nothing to expand—treat as a single block
    if "<|im_start|>" not in content:
        return [{"role": default_role, "content": content.strip()}]
    
    pieces = []
    
    # 1) Grab any leading text before the first <|im_start|>
    first_idx = content.find("<|im_start|>")
    leading = content[:first_idx].strip()
    if leading:
        pieces.append({"role": default_role, "content": leading})
    
    pattern = r"<\|im_start\|>([^\n]+)\n(.*?)<\|im_end\|>"
    blocks = re.findall(pattern, content, flags=re.DOTALL)
    
    for raw_role, block_content in blocks:
        role = raw_role.strip()
        text = block_content.strip()
        # if text:  # skip empty blocks
            # pieces.append({"role": role, "content": text})
        pieces.append({"role": role, "content": text})
    
    return pieces


def convert_to_multiturn(messages):
    """
    Given a list of dicts like:
       [{"role": "user", "content": "..."},
        {"role": "assistant", "content": "... possibly with <|im_start|> ... <|im_end|> ..."}]
    return a new list with multi-turn expansions extracted.
    """
    new_messages = []
    
    for msg in messages:
        if msg["role"] != "assistant":
            # Keep non-assistant messages as-is
            new_messages.append(msg)
        else:
            # Parse out any embedded MPT blocks from assistant messages
            expanded = parse_embedded_mpt_blocks(msg["content"], default_role="assistant")
            new_messages.extend(expanded)
    
    return new_messages



# "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. 
# You should choose the assistant that follows the user's instructions and answers the user's question better. Y
# our evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, 
# and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. 
# Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
# Do not allow the length of the responses to influence your evaluation. 
# Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, 
# output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant 
# B is better, and \"[[C]]\" for a tie.", 
# "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]""

prompt_template_old = [
    {
        'role': 'system',
        'content': (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
            "to the user question displayed below. "
            "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
            "Then, compare the following two conversations between the user and the AI assistants, and provide your evaluation "
            "according to the rubric items. Ensure that the order in which the responses are presented does not influence your decision, and do not "
            "let response length or assistant names affect your evaluation. Be as objective as possible. "
            "Enclose your complete evaluation explanation and final verdict within <eval> and </eval> tags. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Assistant A is better, '[[B]]' if Assistant B is better."
            "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here according to the rubric items</eval>\n\n<answer>[[A/B]]</answer>"
        )
    },
    {
        'role': 'user',
        'content': (
            "[The following is the conversation between Assistant A and the user]\n{conversation1}\n\n"
            "[The following is the conversation between Assistant B and the user]\n{conversation2}"
        )
    }
]

prompt_template_single = [
    {
        'role': 'system',
        'content': (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI chatbots "
            "to the client question displayed below. "
            "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
            "Then, compare the following two conversations between the client and the AI chatbots, and provide your evaluation "
            "according to the rubric items. Ensure that the order in which the responses are presented does not influence your decision, and do not "
            "let response length or chatbot names affect your evaluation. Be as objective as possible. "
            "Enclose your complete evaluation explanation and final verdict within <eval> and </eval> tags. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Chatbot A is better, '[[B]]' if Chatbot B is better."
            "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here according to the rubric items</eval>\n\n<answer>[[A/B]]</answer>"
        )
    },
    {
        'role': 'user',
        'content': (
            "[Client Question]\n{question}\n\n[The Start of Chatbot A's Answer]\n{answer_a}\n[The End of Chatbot A's Answer]\n\n"
            "[The Start of Chatbot B's Answer]\n{answer_b}\n[The End of Chatbot B's Answer]"
        )
    }
]


prompt_template_multi = [
    {
        'role': 'system',
        'content': (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI chatbots "
            "to the client question displayed below. "
            "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
            "Then, compare the following two conversations between the client and the AI chatbots, and provide your evaluation "
            "according to the rubric items. Ensure that the order in which the responses are presented does not influence your decision, and do not "
            "let response length or chatbot names affect your evaluation. Be as objective as possible. "
            "Enclose your complete evaluation explanation and final verdict within <eval> and </eval> tags. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Chatbot A is better, '[[B]]' if Chatbot B is better."
            "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here according to the rubric items</eval>\n\n<answer>[[A/B]]</answer>"
        )
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
            "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
        )
    }
]


file_path = "/shared/nas2/xiusic/Rubric-RM/document_guideline/model_spec_chat.txt"

with open(file_path, "r", encoding="utf-8") as file:
    guideline_document = file.read()




prompt_template_guideline_single = [
    {
        'role': 'system',
        'content': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
            "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "
            "Enclose your detailed evaluation in <eval> and </eval>. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Assistant A is better, '[[B]]' if Assistant B is better."
            "i.e., <eval>detailed evaluation here</eval>\n\n<answer>[[A/B]]</answer>.\n"
            "Use evidence from the assistant's responses to guide your evaluation. You can apply different strategies based on the type of question:\n"
            "- For reasoning questions (math, coding), your evaluation should focus on correctness, intermediate steps, key details, and overall helpfulness.\n"
            f"- For non-reasoning questions (chat, safety), your evaluation should follow the guidelines below:\n\n{guideline_document}",
    },
    {
        'role': 'user',
        'content': "[User Question]\n{question}\n\n[The Start of Assistant A's Response]\n{answer_a}\n[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n{answer_b}\n[The End of Assistant B's Response]"
    }
]

prompt_template_guideline_multi = [
    {
        'role': 'system',
        'content': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
            "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "
            "Enclose your detailed evaluation in <eval> and </eval>. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Assistant A is better, '[[B]]' if Assistant B is better."
            "i.e., <eval>detailed evaluation here</eval>\n\n<answer>[[A/B]]</answer>.\n"
            "Use evidence from the assistant's responses to guide your evaluation. You can apply different strategies based on the type of question:\n"
            "- For reasoning questions (math, coding), your evaluation should focus on correctness, intermediate steps, key details, and overall helpfulness.\n"
            f"- For non-reasoning questions (chat, safety), your evaluation should follow the guidelines below:\n\n{guideline_document}",
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Assistant A and the User]\n{conversation_1}\n[The End of the Conversation between Assistant A and the User]\n\n"
            "[The Start of the Conversation between Assistant B and the User]\n{conversation_2}\n[The End of the Conversation between Assistant B and the User]"
        )
    }
]

CURRENT_NUM = 0
CURRENT_NUM_EASY, CURRENT_NUM_HARD = 0, 0


def substitute_im_tokens(text, start_sub="new_turn_start", end_sub="new_turn_end"):
    """
    Replaces every occurrence of 'im_start' and 'im_end' in the string
    with the specified start_sub and end_sub tokens respectively.

    :param text: The original text
    :param start_sub: The token to replace 'im_start'
    :param end_sub: The token to replace 'im_end'
    :return: A new string with the substitutions made
    """
    new_text = text.replace("im_start", start_sub).replace("im_end", end_sub)
    return new_text



def convert_sky_data_single(input):
    # Input is simply the chosen/rejected 
    # if not (len(input) == 2 and input[0]['role'] == 'user' and input[1]['role'] == "assistant"):
    #     print(input)
    assert len(input) == 2 and input[0]['role'] == 'user' and input[1]['role'] == "assistant"

    question = input[0]['content']
    answer = input[1]['content']
    return question, answer 

def convert_sky_data_single(input):
    # Input is simply the chosen/rejected 
    # if not (len(input) == 2 and input[0]['role'] == 'user' and input[1]['role'] == "assistant"):
    #     print(input)
    assert len(input) == 2 and input[0]['role'] == 'user' and input[1]['role'] == "assistant"

    question = input[0]['content']
    answer = input[1]['content']
    return question, answer 

def convert_sky_data_multi(input, assistant_name="Assistant"):
    result_parts = []
    for entry in input:
        role = entry['role']
        assert role == 'assistant' or role == 'user'
        content = entry['content']

        if role == "assistant":
            role = assistant_name 
        elif role == "user":
            role = "User"
        else:
            raise NotImplementedError()
        result_parts.append(f"{role}: {content}")

    result_string = "\n".join(result_parts)
    return result_string



def collect_Inf_ORM_with_magpie_ultra():
    global CURRENT_NUM 
    ds = load_dataset("infly/INF-ORM-Preference-Magnitude-80K")

    context_messages = []
    winner = []
    score = []

    num_single, num = 0, 0
    for item in ds['train']:
        if item['source'] != "magpie_ultra":

            input_chosen = item['chosen'] 
            input_rej = item['rejected']
            magnitude = item['magnitude']
            if magnitude == 10:
                curr_s  = 0.5
            elif magnitude == 3:
                curr_s  = 0.8
            elif magnitude == 1:
                curr_s  = 1.0
            else:
                raise NotImplementedError("Error")
            curr_s = str(curr_s)
            
            num += 1
            if (len(input_chosen) == 2 and input_chosen[0]['role'] == 'user' and input_chosen[1]['role'] == "assistant") \
                and (len(input_rej) == 2 and input_rej[0]['role'] == 'user' and input_rej[1]['role'] == "assistant"):
                single = True
                num_single += 1  
            else:
                single = False 

            if single:
                question_chosen, answer_chosen = convert_sky_data_single(item['chosen'])
                question_rej, answer_rej = convert_sky_data_single(item['rejected'])
                assert question_chosen == question_rej

                if CURRENT_NUM % 2 == 0:
                    answer_a = answer_chosen
                    answer_b = answer_rej
                    winner.append(('model_a', curr_s))
                else:
                    answer_a = answer_rej
                    answer_b = answer_chosen
                    winner.append(('model_b', curr_s)) 
                
                curr_prompt = copy.deepcopy(prompt_template_guideline_single) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    question=question_chosen,
                    answer_a=answer_a,
                    answer_b=answer_b
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

            else:
                if CURRENT_NUM % 2 == 0:
                    conversation_1 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot B")
                    winner.append(('model_a', curr_s))
                else:
                    conversation_1 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot B")
                    winner.append(('model_b', curr_s)) 
                
                curr_prompt = copy.deepcopy(prompt_template_guideline_multi) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    conversation_1=conversation_1,
                    conversation_2=conversation_2,
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 
        else:
            input_chosen = item['chosen'] 
            input_rej = item['rejected']
            magnitude = item['magnitude']
            if magnitude == 10:
                curr_s  = 0.5
            elif magnitude == 3:
                curr_s  = 0.8
            elif magnitude == 1:
                curr_s  = 1.0 
            else:
                raise NotImplementedError("Error with curr_s")

            curr_s = str(curr_s)

            num_single += 1 
            num += 1
            assert (len(input_chosen) == 2 and input_chosen[0]['role'] == 'user' and input_chosen[1]['role'] == "assistant") \
                and (len(input_rej) == 2 and input_rej[0]['role'] == 'user' and input_rej[1]['role'] == "assistant")
            
            question_chosen, answer_chosen = convert_sky_data_single(item['chosen'])
            question_rej, answer_rej = convert_sky_data_single(item['rejected'])

            answer_chosen, answer_rej = substitute_im_tokens(answer_chosen), substitute_im_tokens(answer_rej)
            assert question_chosen == question_rej 

            if CURRENT_NUM % 2 == 0:
                conversation_1 = f"User: {question_chosen}\nAssistant A: {answer_chosen}"
                conversation_2 = f"User: {question_rej}\nAssistant B: {answer_rej}"
                winner.append(('model_a', curr_s))
            else:
                conversation_1 = f"User: {question_rej}\nAssistant A: {answer_rej}"
                conversation_2 = f"User: {question_chosen}\nAssistant B: {answer_chosen}"
                winner.append(('model_b', curr_s))

            curr_prompt = copy.deepcopy(prompt_template_guideline_multi) 
            curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                conversation_1=conversation_1,
                conversation_2=conversation_2,
            )
            context_messages.append(curr_prompt) 
            CURRENT_NUM += 1

    print(f"Num_single {num_single}, total num: {num}")
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })
    return dataset 

def collect_Inf_ORM_without_magpie_ultra():
    global CURRENT_NUM 
    ds = load_dataset("infly/INF-ORM-Preference-Magnitude-80K")

    context_messages = []
    winner = []
    score = []

    num_single, num = 0, 0
    for item in ds['train']:
        if item['source'] != "magpie_ultra":

            input_chosen = item['chosen'] 
            input_rej = item['rejected']
            magnitude = item['magnitude']
            if magnitude == 10:
                curr_s  = 0.5
            elif magnitude == 3:
                curr_s  = 0.8
            elif magnitude == 1:
                curr_s  = 1.0
            else:
                raise NotImplementedError("Error")
            curr_s = str(curr_s)
            
            num += 1
            if (len(input_chosen) == 2 and input_chosen[0]['role'] == 'user' and input_chosen[1]['role'] == "assistant") \
                and (len(input_rej) == 2 and input_rej[0]['role'] == 'user' and input_rej[1]['role'] == "assistant"):
                single = True
                num_single += 1  
            else:
                single = False 

            if single:
                question_chosen, answer_chosen = convert_sky_data_single(item['chosen'])
                question_rej, answer_rej = convert_sky_data_single(item['rejected'])
                assert question_chosen == question_rej

                if CURRENT_NUM % 2 == 0:
                    answer_a = answer_chosen
                    answer_b = answer_rej
                    winner.append(('model_a', curr_s))
                else:
                    answer_a = answer_rej
                    answer_b = answer_chosen
                    winner.append(('model_b', curr_s)) 
                
                curr_prompt = copy.deepcopy(prompt_template_guideline_single) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    question=question_chosen,
                    answer_a=answer_a,
                    answer_b=answer_b
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

            else:
                if CURRENT_NUM % 2 == 0:
                    conversation_1 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot B")
                    winner.append(('model_a', curr_s))
                else:
                    conversation_1 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot B")
                    winner.append(('model_b', curr_s)) 
                
                curr_prompt = copy.deepcopy(prompt_template_guideline_multi) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    conversation_1=conversation_1,
                    conversation_2=conversation_2,
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

    print(f"Num_single {num_single}, total num: {num}")
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })
    return dataset 

            


                




def collect_skywork_reward():
    global CURRENT_NUM 
    ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

    context_messages = []
    winner = []

    num_single, num = 0, 0
    for item in ds['train']:
        if item['source'] != "magpie_ultra":
            
            input_chosen = item['chosen'] 
            input_rej = item['rejected']
            num += 1
            if (len(input_chosen) == 2 and input_chosen[0]['role'] == 'user' and input_chosen[1]['role'] == "assistant") \
                and (len(input_rej) == 2 and input_rej[0]['role'] == 'user' and input_rej[1]['role'] == "assistant"):
                single = True
                num_single += 1  
            else:
                single = False 


            if single:
                question_chosen, answer_chosen = convert_sky_data_single(item['chosen'])
                question_rej, answer_rej = convert_sky_data_single(item['rejected'])
                assert question_chosen == question_rej

                if CURRENT_NUM % 2 == 0:
                    answer_a = answer_chosen
                    answer_b = answer_rej
                    winner.append('model_a')
                else:
                    answer_a = answer_rej
                    answer_b = answer_chosen
                    winner.append('model_b') 
                
                curr_prompt = copy.deepcopy(prompt_template_single) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    question=question_chosen,
                    answer_a=answer_a,
                    answer_b=answer_b
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

            else:
                # conversation_chosen = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot")
                # conversation_rej = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot")

                if CURRENT_NUM % 2 == 0:
                    conversation_1 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot B")
                    winner.append('model_a')
                else:
                    conversation_1 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot B")
                    winner.append('model_b') 
                
                curr_prompt = copy.deepcopy(prompt_template_multi) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    conversation_1=conversation_1,
                    conversation_2=conversation_2,
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

    print(f"Num_single {num_single}, total num: {num}")
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset 


def collect_code_data():
    global CURRENT_NUM 
    dataset = load_dataset("Vezora/Code-Preference-Pairs", split="train")  # 'train' split as an example
    shuffled = dataset.shuffle(seed=42)
    subset_15k = shuffled.select(range(2_500))
    remainder = shuffled.select(range(2_500, len(shuffled)))

    context_messages = []
    winner = []

    for item in subset_15k:
        question = item['input']
        answer_chosen = item['accepted']
        answer_rej = item['rejected']

        if CURRENT_NUM % 2 == 0:
            answer_a = answer_chosen
            answer_b = answer_rej
            winner.append('model_a')
        else:
            answer_a = answer_rej
            answer_b = answer_chosen
            winner.append('model_b')
        

        curr_prompt = copy.deepcopy(prompt_template_single) 
        curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

        context_messages.append(curr_prompt)
        CURRENT_NUM += 1 
    
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset, remainder

def collect_code_data_guideline():
    global CURRENT_NUM 
    dataset = load_dataset("Vezora/Code-Preference-Pairs", split="train")  # 'train' split as an example
    shuffled = dataset.shuffle(seed=42)
    subset_15k = shuffled.select(range(3000))
    remainder = shuffled.select(range(3000, len(shuffled)))

    context_messages = []
    winner = []

    for item in subset_15k:
        question = item['input']
        answer_chosen = item['accepted']
        answer_rej = item['rejected']

        curr_s = 1.0
        curr_s = str(curr_s) 

        if CURRENT_NUM % 2 == 0:
            answer_a = answer_chosen
            answer_b = answer_rej
            winner.append(('model_a', curr_s))
        else:
            answer_a = answer_rej
            answer_b = answer_chosen
            winner.append(('model_b', curr_s))
        

        curr_prompt = copy.deepcopy(prompt_template_guideline_single) 
        curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

        context_messages.append(curr_prompt)
        CURRENT_NUM += 1 
    
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset, remainder
 

def collect_orm_all():
    ds_orm = collect_Inf_ORM_with_magpie_ultra()
    ds_orm.push_to_hub("gaotang/entire_orm_guideline")


def collect_no_imstart_orm_withcode():
    ds_orm = collect_Inf_ORM_without_magpie_ultra() 
    ds_code, ds_code_remainder = collect_code_data_guideline() 
    ds_orm_code = concatenate_datasets([ds_orm, ds_code])
    ds_orm_code.push_to_hub("gaotang/filtered_orm_guideline")


# def new_dataset_sky():
#     ds_sky = collect_skywork_reward()
#     ds_sky.push_to_hub("gaotang/skywork-v02-filtered")
#     print("Sky pushed")
#     ds_code, ds_code_remainder = collect_code_data()
#     ds_math = collect_math_data()
#     ds_sky_code = concatenate_datasets([ds_sky, ds_code])
#     ds_sky_code.push_to_hub("gaotang/sky_v02_filtered_2_5kcode_sky_code")
#     print("Sky code pushed")

#     ds_sky_code_math = concatenate_datasets([ds_math, ds_code, ds_sky])
#     ds_sky_code_math.push_to_hub("gaotang/sky_v02_filtered_2_5kcode_18kmath_math_code_sky")
#     print("Sky code math pushed")
#     ds_code_remainder.push_to_hub("gaotang/code_reminder_2_5k")
#     print("Code remainder pushed")

    # print(Counter(ds["winner"]))


if __name__ == '__main__':
    # new_dataset_sky()
    # collect_orm_all()
    collect_no_imstart_orm_withcode()
    