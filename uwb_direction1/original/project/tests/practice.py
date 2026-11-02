import torch
import numpy as np


subj_res = [
    {
        "closed_acc": 0.8,
        "head1": {
            "acc": 0.7,
            "f1": 0.6
        }
    },
    {
        "closed_acc": 0.9,
        "head1": {
            "acc": 0.8,
            "f1": 0.7
        }
    }
]
def _avg_results(subj_res):
    heads = []
    for k in subj_res[0]:
        if k != 'closed_acc':
            heads.append(k)

    closed_acc_values=[]
    for r in subj_res:
        closed_acc_values.append(r['closed_acc'])

    out = {
        'closed_acc': float(np.mean(closed_acc_values))
    }

    for head in heads:
       out[head]={}
       
        