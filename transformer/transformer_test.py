import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from transformer import EmbeddingWithLearnedPositionalEncoding, EmbeddingWithPositionalEncoding, TransformerLayer, Encoder, Decoder, EncoderDecoder


class Seq2seqModel(nn.Module):
    def __init__(self, dim, src_n_vocab, n_encod_layer, tgt_n_vocab, n_decode_layer, 
                 max_len=512):
        self.src_emb = EmbeddingWithPositionalEncoding(dim, src_n_vocab, max_len)
        self.tgt_emb = EmbeddingWithLearnedPositionalEncoding(dim, tgt_n_vocab, max_len)

        enc_layer = TransformerLayer(
            dim, 
            MultiHeadAttention(6, dim, 0.1),
            None, 
            nn.Linear(dim, dim),
            0.1
        )
        self.encoder = Encoder(enc_layer, n_encod_layer)

        dec_layer = TransformerLayer(
            dim, 
            MultiHeadAttention(6, dim, 0.1),
            MultiHeadAttention(6, dim, 0.1),
            nn.Linear(dim, dim), 
            0.1
        )
        self.decoder = Decoder(dec_layer, n_decode_layer)

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