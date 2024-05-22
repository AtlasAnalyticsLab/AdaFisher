import numpy as np
import torch
import torch.nn.functional as F
import os

from torch.utils.data import Dataset


class Wikitext2(Dataset):
    """Wikitext-2 dataset.

    This is a PyTorch dataset for the pre-processed Wikitext-2 dataset,
    containing about 2M words. This dataset will produce vectors of tokens
    of fixed size max_length, possibly zero-padded.

    Parameters
    ----------
    root (`string`)
        The root folder containing the `wiki.*.npz` files.

    split (`string` in (`train`, `validation`, `test`))
        The split to use for the dataset.

    break_mode (`string` in (`none`, `lines`, `complete`))
        The mode for breaking down the dataset into sequences. The different
        modes are better explained with an example (with max_length=6)

        Data:
            Recurrent Neural Networks
            Transformer Networks
            Optimization methods for training neural networks
            Regularization of neural networks

        - `none`
            [Recurrent, Neural, Networks, Transformer, Networks, Optimization]
            [methods, for, training, neural, networks, Regularization]
            [of, neural, networks, <pad>, <pad>, <pad>]

        - `lines`
            [Recurrent, Neural, Networks, <pad>, <pad>, <pad>]
            [Transformer, Networks, <pad>, <pad>, <pad>, <pad>]
            [Optimization, methods, for, training, neural, networks]
            [Regularization, of, neural, networks, <pad>, <pad>]

        - `complete`
            [Recurrent, Neural, Networks, Transformer, Networks, <pad>]
            [Optimization, methods, for, training, neural, networks]
            [Regularization, of, neural, networks, <pad>, <pad>]

        For `lines` and `complete`, lines that are longer than `max_length`
        are further broken down into sequences of at most `max_length` (the
        process is similar to `break_mode=none`).

    max_length (`int`)
        The maximal length of a sequence, and the size of all the vectors
        produced by this PyTorch dataset (possibly zero-padded).

    min_length (`int`)
        The minimal length of a sequence. Sequences smaller than this number
        will be dropped from the dataset.
    """

    def __init__(
        self, root, split="train", break_mode="none", max_length=256, min_length=1
    ):

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                'Unknown split "{0}" for Wikitext-2. The split '
                "must be one of {train, validation, test}.".format(split)
            )
        if break_mode not in ["none", "lines", "complete"]:
            raise ValueError(
                'Unknown break_mode "{0}" for Wikitext-2. The '
                "break_mode must be one of {none, lines, complete}.".format(break_mode)
            )

        self.root = os.path.expanduser(root)
        self.split = split
        self.break_mode = break_mode
        self.max_length = max_length
        self.min_length = min_length

        # Load the dataset
        filename = os.path.join(self.root, "wiki.{0}.npz".format(split))
        with open(filename, "rb") as f:
            data = np.load(f)
            self._tokens = torch.from_numpy(data["tokens"].astype(np.int64))
            self._sizes = tuple(data["sizes"])

        # Create the examples depending on the break mode
        if self.break_mode == "none":
            indices = self._get_split_indices(self.num_tokens)
        elif self.break_mode == "lines":
            indices = self._get_split_indices_lines()
        elif self.break_mode == "complete":
            indices = self._get_split_indices_complete()

        self._examples = [
            self._tokens[start : start + length]
            for (start, length) in indices
            if length > self.min_length
        ]

    def __getitem__(self, index):
        example = self._examples[index]
        example_size = example.numel() - 1

        if example_size < self.max_length:
            # Zero-padding of shorter sequences
            remainder = self.max_length - example_size
            source_tokens = F.pad(example[:-1], (0, remainder))
            target_tokens = F.pad(example[1:], (0, remainder))
            mask = torch.zeros(self.max_length, dtype=torch.float)
            mask[:example_size] = 1.0
        else:
            source_tokens, target_tokens = example[:-1], example[1:]
            mask = torch.ones(self.max_length, dtype=torch.float)

        return {"source": source_tokens, "target": target_tokens, "mask": mask}

    def __len__(self):
        return len(self._examples)

    @property
    def num_lines(self):
        return len(self._sizes) + 1

    @property
    def num_tokens(self):
        return self._tokens.numel()

    def _get_split_indices(self, num_tokens):
        num_chunks = (num_tokens - 1) // self.max_length
        indices = [
            (i * self.max_length, self.max_length + 1) for i in range(num_chunks)
        ]
        full_chunks = num_chunks * self.max_length + (num_chunks > 0)
        remainder = num_tokens - full_chunks
        if remainder:
            indices.append((full_chunks, remainder + (num_chunks > 0)))
        return indices

    def _get_split_indices_lines(self):
        indices, start = [], 0
        for size in self._sizes:
            indices.extend(
                [
                    (start + offset, length)
                    for (offset, length) in self._get_split_indices(size)
                ]
            )
            start += size
        return indices

    def _get_split_indices_complete(self):
        # Split according to lines first, and filter short lines
        line_indices = [
            index
            for index in self._get_split_indices_lines()
            if index[1] > self.min_length
        ]
        # Loop over these indices, and merge lines together when
        # they are shorter than the maximum length
        indices, start, length = [], 0, 0
        for index in line_indices:
            if length + index[1] > self.max_length + 1:
                indices.append((start, length))
                start, length = index[0], 0
            length += index[1]

        if length > 0:
            indices.append((start, length))

        return indices
