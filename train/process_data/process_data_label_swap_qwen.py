import copy
from datasets import load_dataset, concatenate_datasets, Dataset


# "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. 
# You should choose the assistant that follows the user's instructions and answers the user's question better. Y
# our evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, 
# and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]""
from collections import Counter

prompt_template = [
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


CURRENT_NUM = 0
CURRENT_NUM_EASY, CURRENT_NUM_HARD = 0, 0

def substitute_tags(input_string):
    output = copy.deepcopy(input_string)
    output = output.replace("<|im_start|>", f"<|begin_of_text|>")
    output = output.replace("<|im_end|>", f"<|end_of_text|>")
    return output

def collect_skywork_reward(replace_special=False):
    global CURRENT_NUM 
    ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

    def reformat_conversation(conv, assistant_name):
        conversations = []
        for item in conv:
            if item['role'] == 'user':
                content = item['content']
                if replace_special:
                    content = substitute_tags(input_string=content)
                conversations.append('User' + ": " + content)
            else:
                content = item['content']
                if replace_special:
                    content = substitute_tags(input_string=content)
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
    ds = collect_skywork_reward(replace_special=True)
    # ds.save_to_disk('/shared/nas2/xiusic/gaotang/data/sky_v02')
    ds.push_to_hub("gaotang/sky_v02_processed_qwen")
    print(Counter(ds["winner"]))


if __name__ == '__main__':
    new_dataset_sky()
    