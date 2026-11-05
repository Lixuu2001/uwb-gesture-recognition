"""Extract the two unmodified pinned radar author helpers without importing its training module."""
from pathlib import Path
import ast
import torch
from torch.nn import functional as F
B=Path(__file__).resolve().parent
def author_functions():
    tree = ast.parse((B / 'author/radar/train.py').read_text())
    wanted = ['_apply_cutmix', '_mixup_ce_loss']
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == 2
    env = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'unmodified_pinned_radar_author_helpers', 'exec'), env)
    return env
