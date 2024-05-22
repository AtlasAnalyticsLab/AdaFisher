import torch
import torch.nn as nn
import numpy as np

class GPT1Embedding(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        sequence_length,
        _tokens_embedding_weight=None,
        _positional_embedding_weight=None,
    ):

        super(GPT1Embedding, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length

        self.tokens = nn.Embedding(
            vocabulary_size,
            embedding_size,
            padding_idx=0,
            _weight=_tokens_embedding_weight,
        )
        self.position = nn.Embedding(
            sequence_length, embedding_size, _weight=_positional_embedding_weight
        )

    def forward(self, tokens, positions):
        """Embedding module for GPT-1.

        This module combines token and positional embeddings, to return the
        embeddings from a sequence of input tokens and positions.
        See Lecture 06, slides 30-32.

        Parameters
        ----------
        tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences. All the tokens
            must be integers in [0, vocabulary_size).

        positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The tensor containing the position indices in the sequence. All
            the positions must be integers in [0, sequence_length)

        Returns
        -------
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_size)`)
            The tensor containing the embeddings. For example, `embeddings[0, 2]`
            is the embedding vector for the token in 3rd position (index 2)
            of the 1st sequence in the batch (index 0).
        """
        if torch.any(positions >= self.sequence_length):
            raise RuntimeError(
                "Some position indices are larger than the " "maximum sequence length."
            )

        if torch.any(tokens >= self.vocabulary_size):
            raise RuntimeError(
                "Some tokens are larger than the size of " "the vocabulary."
            )

        token_embeddings = self.tokens(tokens)
        position_embeddings = self.position(positions)
        return token_embeddings + position_embeddings

    @classmethod
    def load_embeddings_from(cls, filename):
        # Load the embeddings from filename
        with open(filename, "rb") as f:
            embeddings = np.load(f)
            tokens_weight = torch.from_numpy(embeddings["tokens"])
            positional_weight = torch.from_numpy(embeddings["position"])

        vocabulary_size, embedding_size = tokens_weight.shape
        sequence_length = positional_weight.size(0)
        return cls(
            vocabulary_size,
            embedding_size,
            sequence_length,
            _tokens_embedding_weight=tokens_weight,
            _positional_embedding_weight=positional_weight,
        )
