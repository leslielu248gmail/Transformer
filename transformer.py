import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size heads):
    super(MultiHeadAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (
      self.head_dim * heads == embed_size
    ), "Embedding size needs to be divisible by heads"

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

  def forward(self, values, keys, query, mask):
    N = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # Split the embedding into self.heads different pieces
    values = values.reshape(N, value_len, self.heads, self.heads_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = query.reshape(N, query_len, self.heads, self.head_dim)

    attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
    if mask is not None:
      attention = attention.masked_fill(mask==0, float("-1e20"))
    attention = F.softmax(attention / math.sqrt(self.head_dim), dim=3)

    out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(
      N, query_len, self.heads * self.head_dim
    )
    out = self.fc_out(out)
    return out


class PositionalEncoding(nn.Module):
  def __init__(self, embed_size, max_len, device):
    super(PositionalEncoding, self).__init__()
    self.encoding = torch.zeros(max_len, embed_size).to(device)
    self.encoding.requires_grad = False

    pos = torch.agrange(0, max_len).unsqueeze(1).float().to(device)
    _2i = torch.arange(0, embed_size, step=2).float(0).to(device)

    self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
    self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))

  def forward(self, x):
    seq_len = x.shape[1]
    x = x + self.encoding[:seq_len, :]
    return x


class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = MultiHeadAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion * embed_size, embed_size)
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, value, key, query, mask):
    attention = self.attention(value, key, query, mask)

    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)

    out = self.dropout(self.norm2(forward + x))
    return out


class Encoder(nn.Module):
  def __init__(self,
               src_vocab_size,
               embed_size,
               num_layers,
               heads,
               device,
               forward_expansion,
               dropout,
               max_length):
    
    super(Encoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.positional_encoding = PositionalEncoding(embed_size, max_length, device)
    self.layers = nn.ModuleList([
      TransformerBlock(
        embed_size,
        heads,
        dropout = dropout,
        forward_expansion = forward_expansion
      )
      for _ in range(num_layers)
    ])
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    out = self.dropout(self.word_embedding(x))
    out = self.positional_encoding(out)

    for layer in self.layers:
      out = layer(out, out, out, mask)

    return out


def DecoderBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout, device):
    super(DecoderBlock, self).__init__()
    self.attention = MultiHeadAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(
      embed_size, heads, dropout, forward_expansion
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, value, key, src_mask, trg_mask):
    attention = self.attention(x, x, x, trg_mask)
    query = self.dropout(self.norm(attention + x))
    out = self.transformer_block(value, key, query, src_mask)
    return out


class Transformer(nn.Module):
  def __init__(self,
               src_vocab_size,
               trg_vocab_size,
               src_pad_idx,
               trg_pad_idx,
               embed_size=256
               num_layers=6,
               forward_expansion=4,
               heads=8,
               dropout=0,
               device="cuda",
               max_length=100):

    super(Transformer, self).__init__()
    self.encoder Encoder(
      src_vocab_size,
      embed_size,
      num_layers
      heads,
      device,
      forward_expansion,
      dropout,
      max_length
    )
    self.decoder = Decoder(
      trg_vocab_size,
      embed_size,
      num_layers,
      heads,
      forward_expansion,
      dropout,
      device,
      max_length
    )

    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.device = device

  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(self.device)

  def make_trg_mask(self, trg):
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
      N, 1, trg_len, trg_len
    )
    return trg_mask.to(self.device)

  def forward(self, src, trg):
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)

    enc_src = self.encoder(src, src_mask)
    out = self.decoder(trg, enc_src, src_mask, trg_mask)
    return out


                 








