from __future__ import annotations

import torch


def lm_as_judge_match(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
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
