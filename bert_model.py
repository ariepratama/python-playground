from transformera import *
from transformera_embeddings import *


class BertModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 max_position_embeddings: int,
                 pad_token_id: int,
                 n_layers: int,
                 attn_dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 dropout: float = 0.3):
        super().__init__()
        self.embeddings = TransformerEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            dropout=dropout
        )
        self.transformer = Transformer(
            n_layers=n_layers,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """

        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        :return:
            dict(
                last_hidden_state: Tensor(bs, seq_len, dim)
                all_hidden_states: List[Tensor(bs, seq_len, dim)]
                all_attentions: List[Tensor(bs, seq_len, dim)]
            )
        """
        embeddings = self.embeddings(input_ids)
        return self.transformer(embeddings, attn_mask=attention_mask)


class BertForMaskedLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 max_position_embeddings: int,
                 pad_token_id: int,
                 n_layers: int,
                 attn_dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            n_layers=n_layers,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        self.vocab_transform = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.activation = nn.ReLU()
        self.vocab_projector = nn.Linear(hidden_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.LongTensor] = None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # (bs, seq_len, dim)
        last_hidden_state = bert_output["last_hidden_state"]

        logits = self.vocab_transform(last_hidden_state)  # (bs, seq_len, dim)
        logits = self.activation(logits)  # (bs, seq_len, dim)
        logits = self.layer_norm(logits)  # (bs, seq_len, dim)
        logits = self.vocab_projector(logits)  # (bs, seq_len, vocab_size)

        loss = None
        if labels is not None:
            logits = logits.view(-1, logits.size(-1)) # (bs x seq_len, vocab_size)
            loss = self.loss(logits, labels)

        return dict(
            prediction_logits=logits,
            loss=loss,
            hidden_states=bert_output["all_hidden_states"],
            attentions=bert_output["all_attentions"]
        )