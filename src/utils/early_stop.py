# import collections

import numpy as np


class EarlyStop:
    def __init__(self, patience: int = 10, threshold: float = 1e-2) -> None:
        # self.queue = collections.deque([0] * patience, maxlen=patience)
        self.patience = patience
        self.threshold = threshold
        self.wait = 0
        self.best_loss = np.Inf

    def reset(self) -> None:
        self.best_loss = np.Inf
        self.wait = 0

    def __call__(self, train_loss: float) -> bool:
        """
        @monitor: value to monitor for early stopping
                  (e.g. train_loss, test_loss, ...)
        @mode: specify whether you want to maximize or minimize
               relative to @monitor
        """
        if np.less(self.threshold, 0):
            return False
        if train_loss is None:
            return False
        # self.queue.append(train_loss)
        if np.less(train_loss - self.best_loss, -self.threshold):
            self.best_loss = train_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False
