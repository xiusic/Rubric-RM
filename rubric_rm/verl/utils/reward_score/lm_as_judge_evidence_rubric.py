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

pattern = re.compile(
    r'^'
    # 1) <rubric> ... </rubric>, containing <justify> with non-empty text
    r'.*?<rubric>'
    r'(?=.*?<justify>\s*\S.*?\S\s*</justify>)'  # Lookahead ensures <justify> is non-empty
    r'(.*?)'
    r'</rubric>'
    
    # 2) <eval> ... </eval>, must contain references to A and B with non-empty text
    r'.*?<eval>'
    # Chatbot A reference: either <quote_A>...</quote_A> or <summary_A>...</summary_A>, both non-empty
    r'(?=.*?(?:<quote_A>\s*\S.*?\S\s*</quote_A>|<summary_A>\s*\S.*?\S\s*</summary_A>))'
    # Chatbot B reference: either <quote_B>...</quote_B> or <summary_B>...</summary_B>, both non-empty
    r'(?=.*?(?:<quote_B>\s*\S.*?\S\s*</quote_B>|<summary_B>\s*\S.*?\S\s*</summary_B>))'
    r'(.*?)'
    r'</eval>'
    
    # 3) <answer> ... </answer> with non-empty text
    r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
    r'.*$',
    re.DOTALL
)


def validate_string_format(s: str) -> bool:
    """
    Returns True if 's' contains (in order):
      - <rubric> with at least one non-whitespace char
      - <eval> with at least one non-whitespace char
      - <answer> with at least one non-whitespace char
    and potentially anything before/between/after those blocks.
    Otherwise returns False.
    """
    return bool(pattern.match(s))


def lm_as_judge_match(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):

    if validate_string_format(solution_str):
        pred = solution_str.split(
        '</eval>',
        )[-1].strip() if '</eval>' in solution_str else 'error'
        answer = ground_truth
        if answer == 'model_a':
            if '[[A]]' in pred and '[[B]]' not in pred:
                return 1.0
            else:
                return -1.0
        elif answer == 'model_b':
            if '[[B]]' in pred and '[[A]]' not in pred:
                return 1.0
            else:
                return -1.0
        else:
            raise NotImplementedError('Check your dataset label!')
    else:
        return -1.0




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