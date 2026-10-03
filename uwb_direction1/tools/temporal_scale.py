"""One fixed axial dilation intervention; established operation, no novelty claim."""
import torch
from circular_signal import SignalNet


def make_model(arm, seed):
    if arm not in ('shared128', 'time2', 'range2'):
        raise ValueError(arm)
    torch.manual_seed(seed)
    net = SignalNet()
    # Keep every weight, tensor shape, head, normalization, stride and RNG draw.
    # Only the second (stride-one) 3x3 convolution in each block changes spacing.
    dilation = {'shared128': (1,1), 'time2': (2,1), 'range2': (1,2)}[arm]
    for block in list(net.encoder)[:3]:
        block.layers[3].dilation = dilation
        block.layers[3].padding = dilation
    return net
