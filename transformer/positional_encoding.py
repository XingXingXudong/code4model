import math
import torch
import torch.nn as nn


def get_positional_encoding(dim, max_len=5000):
    encodings = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, dim, 2, dtype=torch.float32)
    div_factor = torch.exp(-math.log(10000) * two_i / dim)
    encodings[:, 0::2] = torch.sin(position * div_factor)
    encodings[:, 1::2] = torch.cos(position * div_factor)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class PositionalEncoding(nn.Module):
    def __init__(self, dim_input, dropout_prob, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer("positional_encodings", get_positional_encoding(dim_input, max_len))
    
    def forward(self, x):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    x = torch.randint(1, 10, size=(5, 10)).float()
    print(x.shape)
    print(x)
    pe = PositionalEncoding(10, 0, 10)
    x = pe(x)
    print(x.shape)
    print(x)
    y = get_positional_encoding(10, 5)
    print(y.shape)
    print(y)


