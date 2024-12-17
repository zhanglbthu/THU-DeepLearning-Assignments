import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


def get_masked_attention(attention: Tensor, mask: Tensor, attention_mask: Tensor = None):
    """
    attention: (B, h, T, T)
    mask: (1, 1, T, T)
    attention_mask: (B, T)
    """
    attention = attention.masked_fill(mask == 0, float('-inf'))
    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(attention.dtype).min
        attention = attention + attention_mask
    return attention


def multi_head_self_attention(Q: Tensor,
                              K: Tensor,
                              V: Tensor,
                              mask: Tensor,
                              attention_dropout: nn.Module,
                              attention_mask: Tensor = None):
    """
    Q: (B, h, T, h_dim)
    K: (B, h, T, h_dim)
    V: (B, h, T, h_dim)
    mask: causal mask with shape (1, 1, T, T)
    attention_mask: (optional) padding mask with shape (B, T)
    """
    ############################ Your code here ############################
    # TODO: Implement the multi-head self-attention mechanism
    pass
    ########################################################################


if __name__ == "__main__":
    torch.manual_seed(0)
    Q, K, V = torch.rand(1, 1, 3, 4), torch.rand(1, 1, 3, 4), torch.rand(1, 1, 3, 4)
    abs_error = lambda x, y: torch.abs(x - y).sum().item()

    non_mask = torch.ones(1, 1, 3, 3)
    attention = multi_head_self_attention(Q, K, V, non_mask, nn.Identity())
    attention_gt = torch.tensor([[[
        [0.2388347, 0.3764442, 0.4025784, 0.3575858],
        [0.2417322, 0.3746578, 0.4052466, 0.3076115],
        [0.2409207, 0.3766766, 0.4065247, 0.3363450]
    ]]])
    print("Absolute error without mask:", abs_error(attention, attention_gt))

    mask = torch.tensor([
        [1., 1., 0.],
        [0., 1., 0.],
        [1., 1., 1.]
    ])
    attention = multi_head_self_attention(Q, K, V, mask, nn.Identity())
    attention_gt = torch.tensor([[[
        [0.2798553, 0.4459440, 0.5667919, 0.5700190],
        [0.1852310, 0.3734174, 0.3051000, 0.9320004],
        [0.2409207, 0.3766766, 0.4065247, 0.3363450]
    ]]])
    print("Absolute error with mask:", abs_error(attention, attention_gt))
