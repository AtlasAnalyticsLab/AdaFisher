import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.embeddings import GPT1Embedding


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        layer_size = self.num_heads * self.head_size
        # ==========================
        self.weights_Q = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.weights_V = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.weights_K = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.weights_Y = nn.Linear(in_features=layer_size, out_features=layer_size)
        # ==========================

    def get_attention_weights(self, queries, keys):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. Your attention weights must
        take into account the fact that we have a causal language model, i.e.
        there should be no influence from the future, attention is only
        computed on the past. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. For example, `queries[1, 3, 5]` is the query of
            the 4th head (index 3) for the 6th token (index 5) in the 2nd
            sequence (index 1) in the batch (it is a vector of size `head_size`).

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. For example, `keys[0, 2, 4]` is the key of the
            3rd head (index 2) for the 5th token (index 4) in the 1st sequence
            (index 0) in the batch (it is a vector of size `head_size`).

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch. For example, `attention_weights[1, 3, 5, 7]`
            is the attention weights from the 8th token (index 7) on the 6th
            token (index 5) of the 4th head (index 3) in the 2nd sequence
            (index 1) in the batch. Note that because we have a causal language
            model here, `attention_weights[1, 3, 5, 7] == 0`, since the 8th token
            should not influence on the 6th token (7 > 5).
        """

        # ==========================
        interaction_score = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)

        causal_mask = torch.tril(torch.ones(self.sequence_length, self.sequence_length, device=queries.device).
                                 view(1, 1,  self.sequence_length,  self.sequence_length))

        causal_interaction_score = interaction_score * causal_mask - 1e4 * (1 - causal_mask)
        attention_weights = F.softmax(causal_interaction_score, dim=-1)
        # ==========================
        return attention_weights

    def apply_attention(self, queries, keys, values):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. For example, `queries[1, 3, 5]` is the query of
            the 4th head (index 3) for the 6th token (index 5) in the 2nd
            sequence (index 1) in the batch (it is a vector of size `head_size`).

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. For example, `keys[0, 2, 4]` is the key of the
            3rd head (index 2) for the 5th token (index 4) in the 1st sequence
            (index 0) in the batch (it is a vector of size `head_size`).

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. For example, `values[1, 3, 5]` is the key of the
            4th head (index 3) for the 6th token (index 5) in the 2nd sequence
            (index 1) in the batch (it is a vector of size `head_size`).

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. For
            example, `outputs[0, 2]` contains the output of the attention
            (concatenated for all heads) for the 3rd token (index 2) of the 1st
            sequence in the batch (index 0).
        """

        # ==========================
        A = self.get_attention_weights(queries, keys)
        h = torch.matmul(A, values)
        return self.merge_heads(h)
        # ==========================


    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        # ==========================
        splitted_head = tensor.reshape(tensor.size(0), tensor.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        # ==========================
        return splitted_head

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        B, T, Emb_dim = tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3]
        merged_tensor = tensor.transpose(1, 2).reshape(B, T, Emb_dim)
        # ==========================
        return merged_tensor

    def forward(self, hidden_states):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        # ==========================
        # Q = self.split_heads(torch.matmul(hidden_states, self.weights_Q) + self.bias_Q)
        # K = self.split_heads(torch.matmul(hidden_states, self.weights_K) + self.bias_K)
        # V = self.split_heads(torch.matmul(hidden_states, self.weights_V) + self.bias_V)
        # Y = self.apply_attention(Q, K, V)
        # outputs = torch.matmul(Y, self.weights_Y) + self.bias_Y

        Q = self.split_heads(self.weights_Q(hidden_states))
        K = self.split_heads(self.weights_K(hidden_states))
        V = self.split_heads(self.weights_V(hidden_states))
        Y = self.apply_attention(Q, K, V)
        outputs = self.weights_Y(Y)

        # ==========================
        return outputs


class Block(nn.Module):
    def __init__(self, head_size, mlp_hidden_size, num_heads, sequence_length):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.
        """
        super(Block, self).__init__()
        self.head_size = head_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.hidden_size = num_heads * head_size

        self.attention = MultiHeadedAttention(head_size, num_heads, sequence_length)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, self.hidden_size),
        )
        self.norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states):
        attention_outputs = self.attention(hidden_states)
        attention_outputs = self.norm1(attention_outputs + hidden_states)
        outputs = self.mlp(attention_outputs)
        outputs = self.norm2(outputs + attention_outputs)
        return outputs


class MiniGPT1(nn.Module):
    def __init__(
            self,
            vocabulary_size=40479,
            embedding_size=768,
            sequence_length=256,
            num_heads=12,
            num_layers=4,
            learn_embeddings=False,
            _tokens_embedding_weight=None,
            _positional_embedding_weight=None,
    ):
        super(MiniGPT1, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learn_embeddings = learn_embeddings
        self.head_size = embedding_size // num_heads

        self.embedding = GPT1Embedding(
            vocabulary_size,
            embedding_size,
            sequence_length,
            _tokens_embedding_weight=_tokens_embedding_weight,
            _positional_embedding_weight=_positional_embedding_weight,
        )
        self.layers = nn.ModuleList(
            [
                Block(self.head_size, 4 * embedding_size, num_heads, sequence_length)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(embedding_size, vocabulary_size, bias=False)

        # Tying classifier and embedding weights
        self.classifier.weight = self.embedding.tokens.weight

        # Freeze the embedding weights, depending on learn_embeddings
        self.embedding.requires_grad_(learn_embeddings)

    def get_embeddings(self, inputs):
        """Get the embeddings for some input sequence.

        This function computes the embedding vectors based on the input
        sequence (and the positions of the tokens). See also the module
        `GPT1Embedding` for details about the implementation of
        `self.embedding`.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        Returns
        -------
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_size)`)
            The tensor containing the embeddings. For example, `embeddings[0, 2]`
            is the embedding vector for the token in 3rd position (index 2)
            of the 1st sequence in the batch (index 0).
        """

        # ==========================
        position = torch.arange(0, self.sequence_length, device=inputs.device).unsqueeze(0)
        embeddings = self.embedding(inputs, position)
        # ==========================
        return embeddings

    def forward(self, inputs):
        """Mini GPT-1.

        This is a small version of OpenAI's GPT-1 transformer for (causal)
        language modeling. This module returns for each position in the
        sequence the log-probabilities of the next token.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        Returns
        -------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch. For example, `log_probas[0, 3, 6]`
            corresponds to log p(x_{5} = token_{7} | x_{0:4}) (x_{5} for the word
            after x_{4} at index 3, and token_{7} for index 6) for the 1st sequence
            of the batch (index 0).
        """

        # ==========================
        x = self.get_embeddings(inputs)
        for layer in self.layers:
            x = layer(x)
        outputs = nn.LogSoftmax(dim=-1)(self.classifier(x))
        # ==========================
        return outputs

    def loss(self, log_probas, targets, mask):
        """Loss function.

        This function computes the loss (negative log-likelihood).

        Parameters
        ----------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch.

        targets (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing the target next tokens for all positions in
            each sequence of the batch.

        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing values in {0, 1} only, where the value is 0
            for positions corresponding to padding in the sequence, and 1
            otherwise.

        Returns
        -------
        loss (`torch.FloatTensor` scalar)
            The scalar loss, corresponding to the (mean) negative log-likelihood.
        """

        # ==========================
        loss = nn.NLLLoss(reduction='none')(log_probas.view(-1, log_probas.size(-1)),
                                            targets.view(-1))

        masked_loss = loss * mask.view(-1)
        mean_loss = masked_loss.sum() / mask.sum()
        # ==========================
        return mean_loss

    @classmethod
    def load_embeddings_from(
            cls, filename, num_heads=12, num_layers=4, learn_embeddings=False
    ):
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
            num_heads,
            num_layers,
            learn_embeddings,
            _tokens_embedding_weight=tokens_weight,
            _positional_embedding_weight=positional_weight,
        )
