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
    
    # 2) Extract all <|im_start|>role\n...<|im_end|> blocks
    #    Explanation of the pattern:
    #      - <\|im_start\|> : matches literal "<|im_start|>"
    #      - ([^\n]+)       : role on the same line (captured group #1)
    #      - \n             : newline after the role
    #      - (.*?)          : any text until the next <|im_end|> (captured #2)
    #      - <\|im_end\|>   : matches literal "<|im_end|>"
    #    DOTALL so that '.' includes newlines
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

prompt_template = [
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


CURRENT_NUM = 0
CURRENT_NUM_EASY, CURRENT_NUM_HARD = 0, 0


def convert_sky_data(input):
    # Input is simply the chosen/rejected 
    assert len(input) == 2 and input[0]['role'] == 'user' and input[1]['role'] == "assistant"

    question = input[0]['content']
    answer = input[1]['content']
    return question, answer 

def collect_skywork_reward(replace_special=False):
    global CURRENT_NUM 
    ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

    # 1. we first convert this dataset into appropriate multi-turn format 
    new_sky_data = []
    for item in ds['train']:
        new_item = {
            'chosen': convert_to_multiturn(item['chosen']),
            'rejected': convert_to_multiturn(item['rejected']), 
            'source': item['source']
        }
        new_sky_data.append(new_item)


    def reformat_conversation(conv, assistant_name):
        conversations = []
        for item in conv:
            if item['role'] == 'user':
                content = item['content']
                if replace_special:
                    content = substitute_tags(input_string=content, assistant_name=assistant_name)
                conversations.append('User' + ": " + content)
            else:
                content = item['content']
                if replace_special:
                    content = substitute_tags(input_string=content, assistant_name=assistant_name)
                conversations.append(assistant_name + ": " + content)
        return '\n'.join(conversations)


    context_messages = []
    winner = []

    for item in ds['train']:
        winning_response = item['chosen']
        losing_response = item['rejected']

        if CURRENT_NUM % 2 == 0:
            conversation1 = reformat_conversation(
                conv=winning_response, 
                assistant_name="Assistant A"
            )
            conversation2 = reformat_conversation(
                conv=losing_response,
                assistant_name="Assistant B"
            )
            winner.append('model_a')
        else:
            conversation1 = reformat_conversation(
                conv=losing_response,
                assistant_name="Assistant A"
            )
            conversation2 = reformat_conversation(
                conv=winning_response,
                assistant_name="Assistant B"
            )
            winner.append('model_b') 

        curr_prompt = copy.deepcopy(prompt_template) 
        curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
            conversation1=conversation1,
            conversation2=conversation2
        )

        context_messages.append(curr_prompt)
        CURRENT_NUM += 1 

    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset 
 

def new_dataset_sky():
    ds = collect_skywork_reward()
    ds.push_to_hub("gaotang/sky_v02_processed_llamma3")
    print(Counter(ds["winner"]))


if __name__ == '__main__':
    new_dataset_sky()
    