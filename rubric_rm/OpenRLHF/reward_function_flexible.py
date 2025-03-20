# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    preds = [x.split("</eval>")[-1].strip() if '</eval>' in x else 'error' for x in queries]
    r = []
    for answer, pred in zip(labels, preds):
        if answer == 'model_a':
            if '[[A]]' in pred and '[[B]]' not in pred:
                r.append(torch.tensor([1.0])) 
            else:
                r.append(torch.tensor([-1.0]))    
        elif answer == 'model_b':
            if '[[B]]' in pred and '[[A]]' not in pred:
                r.append(torch.tensor([1.0]))
            else:
                r.append(torch.tensor([-1.0]))
        else:
            raise NotImplementedError("Check your dataset label!")
    r = torch.cat(r)
    return r