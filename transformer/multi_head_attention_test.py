import torch
import torch.nn as nn
from .multi_head_attention import HeadSplit, MultiHeadAttention

import pytest

def test_head_split():
    # input_dim = 10, heads_num = 4, key_dim = 8
    head_split = HeadSplit(10, 4, 8, False)
    x = torch.randn(3, 10)
    y = head_split(x)
    assert list(y.shape) == [3, 4, 8]

    x = torch.randn(5, 3, 10)
    y = head_split(x)
    assert list(y.shape) == [5, 3, 4, 8]

def test_multi_head_attention():
    multi_head_attention = MultiHeadAttention(8, 128)
    query = torch.randn(64, 4, 128)
    key = torch.randn(64, 4, 128)
    value = torch.randn(64, 4, 128)
    y = multi_head_attention(query, key, value)
    assert list(y.shape) == [64, 4, 128]
    assert list(multi_head_attention.attn.shape) == [64, 64, 4, 8]

# different dims
def test_multi_head_attention_diff():
    multi_head_attention = MultiHeadAttention(8, 128)
    query = torch.randn(64, 4, 128)
    key = torch.randn(32, 4, 128)
    value = torch.randn(32, 4, 128)
    y = multi_head_attention(query, key, value)
    assert list(y.shape) == [64, 4, 128] 
    assert list(multi_head_attention.attn.shape) == [64, 32, 4, 8]