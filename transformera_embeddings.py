import torch
from torch import nn


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_position_embeddings: int, pad_token_id: int,
                 dropout: float = 0.3):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids: torch.Tensor):
        """

        :param input_ids: (batch_size, seq_len)
        :return:
        """
        # input_embeds = (batch_size, seq_len, embed_dim)
        input_embeddings = self.word_embeddings(input_ids)
        seq_len = input_embeddings.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)  # (seq_len)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch_size, seq_len)
        position_embeddings = self.position_embeddings(position_ids) # (batch_size, seq_len, embed_dim)

        embeddings = input_embeddings + position_embeddings # (batch_size, seq_len, embed_dim)
        embeddings = self.embed_norm(embeddings) # (batch_size, seq_len, embed_dim)
        embeddings = self.dropout(embeddings)
        return embeddings

