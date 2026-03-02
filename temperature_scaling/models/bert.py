import torch.nn as nn
from .transformer import TransformerBlock  # Assuming you have a TransformerBlock implementation
from .embedding import BERTEmbedding  # Assuming you have a BERTEmbedding implementation

class BERTForTextClassification(nn.Module):
    """
    BERT model for text classification.
    """

    def __init__(self, vocab_size, num_classes=20, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param num_classes: number of output classes for text classification
        :param hidden: BERT model hidden size
        :param n_layers: number of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.num_classes = num_classes

        # Paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # Embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # Multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # Classifier head for text classification
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # Attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len])
        x = x['input_ids']
        mask = (x > 0).unsqueeze(2).repeat(1, 1, x.size(2), 1).repeat(1, self.attn_heads, 1, 1) # SHAPE: torch.Size([8, 1, 512, 512])

        # Embedding the indexed sequence to sequence of vectors
        x = self.embedding(x) # SHAPE: torch.Size([8, 1, 512, 768])

        # Running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        x = x.squeeze(1)
        # Extract the [CLS] token representation for classification
        cls_token_rep = x[:, 0, :]  # Assuming [0] is the [CLS] token position

        # Pass the [CLS] token representation through the classifier head
        logits = self.classifier(cls_token_rep)

        return logits
