from __future__ import annotations

import torch



import re

# Explanation of the core snippet, repeated for each tag:
# <rubric>\s*(\S.*?\S|\S)\s*</rubric>
#
# - \s* allows leading whitespace inside the tag before the real content
# - (\S.*?\S|\S) ensures there's at least one non-whitespace character:
#     - \S.*?\S means there's a non-whitespace at the start and at the end (with anything in-between).
#     - |\S covers the case of a single-character non-whitespace (so it doesn't force two).
# - \s* then allows trailing whitespace (e.g. <rubric>   content   </rubric>)
#
# We wrap these in  .*?  (non-greedy, DOTALL) to allow anything around/between the tags.

# pattern = re.compile(
#     r'^'
#     # Look for <rubric>...</rubric> with some content inside
#     r'.*?<rubric>\s*(\S.*?\S|\S)\s*</rubric>'
#     # Look for <eval>...</eval> containing:
#     #   - either <quote_A>...</quote_A> or <summary_A>...</summary_A>
#     #   - AND either <quote_B>...</quote_B> or <summary_B>...</summary_B>
#     r'.*?<eval>\s*'
#         r'(?=.*?(?:<quote_A>.*?</quote_A>|<summary_A>.*?</summary_A>))'
#         r'(?=.*?(?:<quote_B>.*?</quote_B>|<summary_B>.*?</summary_B>))'
#         r'(.*?)'  # Capture the <eval> content
#     r'\s*</eval>'
#     # Finally look for <answer>...</answer> with some content
#     r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
#     r'.*?$',
#     re.DOTALL
# )

def check_task_format(text: str) -> bool:
    # Pattern for Chat task
    chat_pattern = re.compile(
        r'^'
        r'.*?<type>\s*Chat\s*</type>'
        r'.*?<rubric>'
        r'(?=.*?\S)'  # rubric must not be empty
        r'(?=.*?<justify>.*?\S+.*?</justify>)'
        r'(.*?)'
        r'</rubric>'
        r'.*?<eval>'
        r'(?=.*?(<quote_A>.*?\S+.*?</quote_A>|<summary_A>.*?\S+.*?</summary_A>))'
        r'(?=.*?(<quote_B>.*?\S+.*?</quote_B>|<summary_B>.*?\S+.*?</summary_B>))'
        r'(.*?)'
        r'</eval>'
        r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
        r'.*$',
        re.DOTALL
    )

    # Pattern for Reasoning task
    reasoning_pattern = re.compile(
        r'^'
        r'.*?<type>\s*Reasoning\s*</type>'
        r'.*?<solution>.*?\S+.*?</solution>'
        r'.*?<eval>'
        r'(?=.*?(<quote_A>.*?\S+.*?</quote_A>|<summary_A>.*?\S+.*?</summary_A>))'
        r'(?=.*?(<quote_B>.*?\S+.*?</quote_B>|<summary_B>.*?\S+.*?</summary_B>))'
        r'(.*?)'
        r'</eval>'
        r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
        r'.*$',
        re.DOTALL
    )

    return bool(chat_pattern.match(text)) or bool(reasoning_pattern.match(text))


def format_reward(s: str) -> float:
    if check_task_format(s):
        return 1.0 
    else:
        return 0.0

def answer_reward(solution_str: str, answer: str) ->float:
    pred = solution_str[-40:]

    if answer == 'model_a':
        if '<answer>[[A]]</answer>' in pred and '<answer>[[B]]</answer>' not in pred:
            return 1.0
        else:
            return -1,0
    elif answer == 'model_b':
        if '<answer>[[B]]</answer>' in pred and '<answer>[[A]]</answer>' not in pred:
            return 1.0
        else:
            return -1.0
    else:
        raise NotImplementedError('Check your dataset label!')

def lm_as_judge_match(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    r = answer_reward(solution_str, ground_truth)
    
    return r 




### Strict format here:

# pattern_strict = re.compile(
#     r'^'
#     # optional whitespace, then <rubric> with non-whitespace content
#     r'\s*<rubric>\s*(\S(?:.*?\S)?)\s*</rubric>'
#     # only whitespace allowed between </rubric> and <eval>
#     r'\s*<eval>\s*(\S(?:.*?\S)?)\s*</eval>'
#     # only whitespace allowed between </eval> and <answer>
#     r'\s*<answer>\s*(\S(?:.*?\S)?)\s*</answer>\s*'
#     r'$',
#     re.DOTALL
# )

# def validate_string_format_strict(s: str) -> bool:
#     """
#     Enforces this strict format:
#       [optional whitespace]
#       <rubric> (non-whitespace content) </rubric>
#       [only whitespace allowed in between]
#       <eval>   (non-whitespace content) </eval>
#       [only whitespace allowed in between]
#       <answer> (non-whitespace content) </answer>
#       [optional whitespace]
      
#     'Non-whitespace content' means at least one real character – not just spaces/tabs/newlines.
#     Returns True if 's' matches exactly that structure; otherwise False.
#     """
#     return bool(pattern_strict.match(s))