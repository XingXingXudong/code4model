import torch
import math
import copy
import torch.nn as nn
from torch.nn.functional import margin_ranking_loss

from multi_head_attention import MultiHeadAttention
from positional_encoding import get_positional_encoding

class EmbeddingWithPositionalEncoding(nn.Module):
    """This module give the embedings whit positional embedings added.
    Args:
        dim (int): size of each encodings
        n_vocab (int): size of the dictionary of encodings
        max_len (int): max size of sequences length
    
    Shape:
        input (*) LongTensor of arbitrary shape containing the indices to sequence tokens
        Ouput (*, H) where * is the input shape and H = dim
    """
    def __init__(self, dim, n_vocab, max_len):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim)
        self.dim = dim
        self.register_buffer("positional_encodings", get_positional_encoding(dim, max_len=max_len))

    def forward(self, x):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        # 有部分实现为：self.emb(x) * math.sqrt(self.dim) + pe, 这里之所以是要乘以math.sqrt(self.dim)，原因还是
        # 要保持embedings为0-均值，1-方差的分布，之所以有代码这么实现，就是应为self.emb的分布为0均值，1/self.dim方差的。
        return self.emb(x) * math.sqrt(self.dim) + pe

class EmbeddingWithLearnedPositionalEncoding(nn.Module):
    """This module give the embedings whit positional embedings added.
    Args:
        dim (int): size of each encodings
        n_vocab (int): size of the dictionary of encodings
        max_len (int): max size of sequences length
    
    Shape:
        input (*) LongTensor of arbitrary shape containing the indices to sequence tokens
        Ouput (*, H) where * is the input shape and H = dim
    """
    def __init__(self, dim, n_vocab, max_len):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim)
        self.dim = dim
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, dim), requires_grad=True)

    def forward(self, x):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        # 有部分实现为：self.emb(x) * math.sqrt(self.dim) + pe, 这里之所以是要乘以math.sqrt(self.dim)，原因还是
        # 要保持embedings为0-均值，1-方差的分布，之所以有代码这么实现，就是应为self.emb的分布为0均值，1/self.dim方差的。
        return self.emb(x) * math.sqrt(self.dim) + pe

class TransformerLayer(nn.Module):
    """Tansformer layer, this can act as an encoder layer or a decoder layer.
       Some implementations, including the paper seem to have differences in where the layer-normalization is done. 
       Here we do a layer normalization before attention and feed-forward networks, and add the original residual 
       vectors. Alternative is to do a layer normalization after adding the residuals. But we found this to be less 
       stable when training. We found a detailed discussion about this in the paper On Layer Normalization in the 
       Transformer Architecture.
    Args:
        dim (int): size of token embeding
        self_attn (Module): module of self attention
        src_attn (Module): module of source attention
        feed_forward (Module): module of feed forward 
        dropout_prob (float): the probability of the dropout layer
    """
    def __init__(self, dim, self_attn, src_attn, feed_forward, dropout_prob):
        super().__init__()
        self.dim = dim
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.droput = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([dim])
        if src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([dim])
        self.norm_ff = nn.LayerNorm([dim])

        # where to save input to the feed forward layer
        self.is_save_ff_input = False

    def forward(self, x, mask, src=None, src_mask=None):
        # normalize the vectors befor doing self attention
        z = self.norm_self_attn(x)
        # run through self attention i.e. the key and value are from self
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # add the self attention results
        x = x + self.droput(self_attn)
        # if a source is provided, get result from attention to source. 
        # this is when you have an decoder layer that pays attention to encoder output.
        if src is not None:
            # normalize vectors
            z = self.norm_src_attn(x)
            # attention to source i.e. keys and values are from source
            src_attn = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            # add the source attention result
            x = x + src_attn
        # normalize for feed-forward
        z = self.norm_ff(x)
        # save the input to feed forward layer if sepecified.
        if self.is_save_ff_input:
            self.ff_input = z.clone()

        # passt through the feed-forward network
        ff = self.feed_forward(z)
        # add the feed-forward result back
        x = x + self.droput(ff)

        return x

class Encoder(nn.Module):
    """Transformer Encoder
    Args:
        layer (TransformerLayer): the layer of Transformer
        num_layer (int): layer num of TransformerLayers
    """
    def __init__(self, layer, num_layer):
        super().__init__()
        # make copeis of transformer layers
        # self.layers = nn.ModuleList([copy(layer) for _ in range(num_layer)])
        self.layers = nn.ModuleList([layer for _ in range(num_layer)])
        # final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """Transformer Decoder

    Args:
        layer (TransformerLayer): the layer of Transformer
        num_layer (int): layer num of TransformerLayers
    """
    def __init__(self, layer, num_layer):
        super().__init__()
        # make copeis of transformer layers
        self.layers = nn.ModuleList([copy(layer) for _ in range(num_layer)])
        # final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask, memory, src_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_emb, tgt_emb):
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encode(src, src_mask)
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_emb(tgt), memory, src_mask, tgt_mask)


class Seq2seqModel(nn.Module):
    def __init__(self, dim, src_n_vocab, n_encod_layer, tgt_n_vocab, n_decode_layer, 
                 max_len=512):
        super().__init__()
        self.src_emb = EmbeddingWithPositionalEncoding(dim, src_n_vocab, max_len)
        self.tgt_emb = EmbeddingWithLearnedPositionalEncoding(dim, tgt_n_vocab, max_len)

        self.enc_layer = TransformerLayer(
            dim, 
            MultiHeadAttention(6, dim, 0.1),
            None, 
            nn.Linear(dim, dim),
            0.1
        )
        self.encoder = Encoder(self.enc_layer, n_encod_layer)

        self.dec_layer = TransformerLayer(
            dim, 
            MultiHeadAttention(6, dim, 0.1),
            MultiHeadAttention(6, dim, 0.1),
            nn.Linear(dim, dim), 
            0.1
        )
        self.decoder = Decoder(self.dec_layer, n_decode_layer)

        self.encoder_decoder = EncoderDecoder(
            self.encoder, 
            self.decoder, 
            self.src_emb, 
            self.tgt_emb
        )

    def forward(self, x, tgt, src_mask, tgt_mask):
        return self.encoder_decoder(x, tgt, src_mask, tgt_mask)


if __name__ == '__main__':
    m = Seq2seqModel(100, 2, 200, 2, 128)
    src = torch.Tensor([[1, 2, 1, 3], [4, 10, 4, 1]]).long()
    tgt = torch.Tensor([[1, 2, 1, 3], [4, 10, 4, 1]]).long()
    y = m(src, tgt, None, None)
    print(y)

