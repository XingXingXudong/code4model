import math
import torch
from torch import nn as nn

class HeadSplit(nn.Module):
    """Does as linear transformation and splits the vector into given number of heads for multi-head attention.
    Used to transform `key`, `query` and `value` vectors.
    """
    def __init__(self, dim_input, num_head, dim_key, bias):
        """Linear transformation for head split.

        Args:
            dim_input (int): dimension of input feature
            num_head (int): num of heads
            dim_key (int): dimension of key vector 
            bias (bool, optional): has bias if True. Defaults to True.
        """
        super().__init__()
        self.linear = nn.Linear(dim_input, num_head * dim_key, bias=bias)
        self.num_head = num_head
        self.dim_key = dim_key

    def forward(self, input):
        """Linear transfomation and for head split.

        Args:
            input (Tensor): input tensor. Shape [..., dim_input]

        Returns:
            Tensor: output tensor. Shape [..., num_head, dim_key]
        """
        head_shape = input.shape[:-1]
        out = self.linear(input)
        out = out.view(*head_shape, self.num_head, self.dim_key)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, dim_input, dropout_prob=0.1, bias=True):
        """Computes scaled multi-head attention for given query, key and value vectors.
            `Attention(Q, K, V) = softmax_{seq}(\frac{QK^T}{\sqrt d_{k}})V`

           In simple terms, it finds keys that matches the query, and gets the values of those keys.
           It uses dot-product of query and key as the wieghts of how matching the are. Befor taking the `softmax` the
           dot-products are scaled by `\frac{1}{\sqrt d_k}`. This is done to avoid large dot-product values causing 
           softmax to give very small gradients when `d_k` is large or too small.
        Args:
            num_head (int): the number of heads.
            dim_input (int ): the num of features in the query, key and value vectors.
            dropout_prob (float, optional): dropout probability. Defaults to 0.1.
            bias (bool, optional): has bias. Defaults to True.
        """
        super().__init__()
        self.dim_key = dim_input // num_head
        self.num_head = num_head

        self.query = HeadSplit(dim_input, num_head, self.dim_key, bias=bias)
        self.key = HeadSplit(dim_input, num_head, self.dim_key, bias=bias)
        self.value = HeadSplit(dim_input, num_head, self.dim_key, bias=True)

        # Softmax is calculated along the axis of the sequence(or time)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(dim_input, dim_input)
        self.dropout = nn.Dropout(dropout_prob)

        # why scale? If we assume that the components of Q and K are independent random variables with mean 0 and 
        # variance 1, then their dot product has mean 0 and variance d_k. When d_k is big i.e the the lenght of query is 
        # longer more，the softmax(QK^T) is too big or too small, thus the gradients of softmax(QK^k) is too small for back 
        # propagation.
        self.scale = 1 / math.sqrt(self.dim_key)

        self.attn = None

    def get_score(self, query, key):
        """calculate scores between queries and keys.
           This method can be overridden for other variations like relative attention.
               `QK^T` or S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}`
        Args:
            query (Tensor): query tensor. Shape [seq_len, batch_size, head_num, dim_key]
            key (Tensor): key tensor. Shape [seq_len, batch_size, head_num, dim_key]

        Returns:
            Tensor: the query socre i.e the weight of value. Shape [seq_len, seq_len, batch_size, head_num]
        """
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def forward(self, query, key, value, mask=None):
        """The forward of multi-head attention.
        Args:
            query (Tensor): query tensor. Shape [seq_len, batch_size, dim_feature]
            key (Tensor): key tensor. Shape [seq_len, batch_size, dim_feature]
            value (Tensor): value tensor. Shape [seq_len, batch_size, dim_feature]
            mask (Tensor, optional): mask tensor. mask[i, j, b] indicates wheter for batch b, query at position i has 
            access to key value at position j. Defaults to None. Shape [seq_len, seq_len, batch_size]
        Returns:
            Tensor: forward result. Shape [seq_len, batch_size, dim_feature]
        """
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            # where first dimention is the query dimension. If the query dimension is equal to 1 it will be broadcasted.
            assert mask.shape[0] == 1 or mask.shape[0] == mask.shape[1]
            # same mask applied to all head. Will be broadcasted to [seq_len, seq_len, batch_size, head_num]
            mask = mask.unsqueeze(-1)

        # prepare query, key, and value for attention computation. These will have 
        # shape [seq_len, batch_size, head_num, dim_key]
        query = self.query(query)
        key = self.query(key)
        value = self.value(value)

        # compute attention scores `QK^T`, This gives a tensor of shape [seq_len, seq_len, batch_size, head_num]
        scores = self.get_score(query, key)
        # scale scores by `\sqrt d_k`
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # compute attention 'weight score'
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # multiply the 'weight score' and values，get the attented values
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        
        # save attentions for any other calculations
        self.attn = attn.detach()

        # concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # output layer
        return self.output(x)
