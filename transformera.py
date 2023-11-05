import logging
import math
import sys
from typing import *

import torch
from torch import nn

FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger("transformera")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


def separate_heads(x: torch.Tensor, bs: int, n_heads: int, dim_per_head: int) -> torch.Tensor:
    """ separate seq_len into per head

    :param x:
    :param bs:
    :param n_heads:
    :param dim_per_head:
    :return:
    """
    return x.view(bs, -1, n_heads, dim_per_head).transpose(1, 2)


def unseparate_heads(x: torch.Tensor, bs: int, n_heads: int, dim_per_head: int) -> torch.Tensor:
    return x.transpose(1, 2).contiguous().view(bs, -1, n_heads * dim_per_head)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.q_lin = nn.Linear(in_features=dim, out_features=dim)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim)
        self.attn_weight_dropout = nn.Dropout(p=0.4)

        self.n_heads = n_heads
        self.attention_head_size = self.dim // self.n_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        """

        :param query: torch.Tensor (bs, seq_length, dim)
        :param key: torch.Tensor (bs, seq_length, dim)
        :param value: torch.Tensor (bs, seq_length, dim)
        :param mask: torch.Tensor (bs, seq_length)
        :return:
            weights (bs, n_heads, seq_length, seq_length)
            Attention weights context (bs, seq_length, dim)
        """
        bs, q_length, dim = query.size()
        dim_per_head = self.dim // self.n_heads

        k_length = key.size(1)
        mask_reshp = (bs, 1, 1, k_length)

        # divide each tensors into heads -> (bs, seq_len // n_heads, dim)
        q = separate_heads(query, bs, self.n_heads, dim_per_head)
        k = separate_heads(key, bs, self.n_heads, dim_per_head)
        v = separate_heads(value, bs, self.n_heads, dim_per_head)

        q = q / math.sqrt(dim_per_head)
        # (bs, seq_length // n_head, n_head, dim) x (bs, seq_length // n_head, dim, n_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        m: torch.Tensor = (mask == 0)

        # reshape mask to be (batch_size, 1, 1, sequence_length)
        # then repeat the mask for
        mask = m.view(mask_reshp).expand_as(scores)

        scores = scores.masked_fill(
            mask,
            # fill masked value with the minimum value of the tensor type
            torch.tensor(torch.finfo(scores.dtype).min)
        )

        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.attn_weight_dropout(weights)

        context = torch.matmul(weights, v)
        # switch shape back to (bs, seq_len, dim)
        context = unseparate_heads(context, bs, self.n_heads, dim_per_head)
        out = self.out_lin(context)
        return out, weights


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """ TransformerBlock

    return:
        logits: Tensor(bs, seq_len, dim),
        weightL Tensor(bs, seq_len, dim)
    """
    def __init__(self, attn_dim: int, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(dim=attn_dim, n_heads=n_heads)
        self.attention_norm = nn.LayerNorm(normalized_shape=attn_dim, eps=1e-12)
        self.ffn = FFN(dim=attn_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(normalized_shape=attn_dim, eps=1e-12)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = (bs, seq_len, dim)
        # attention_output = (bs, seq_len, dim)
        attention_output, attention_weight = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask
        )

        attention_output = self.attention_norm(attention_output + x)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_norm(ffn_output)
        return ffn_output, attention_weight


class Transformer(nn.Module):
    """ Transformer

        return:
            logits / hidden_state: Tensor(bs, seq_len, dim)
        """
    def __init__(self, n_layers: int, attn_dim: int, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(self.n_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        hidden_state = x
        all_hidden_states = []
        all_attentions = []
        for transformer_block in self.layers:
            hidden_state, attention_weight = transformer_block(hidden_state, attn_mask=attn_mask)
            all_hidden_states.append(hidden_state)
            all_attentions.append(attention_weight)

        return dict(
            last_hidden_state=hidden_state,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions
        )