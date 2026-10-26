import numpy as np


class DataLoader(object):
    """Итератор по мини-батчам с опциональным шаффлом."""

    def __init__(self, X, y, batch_size: int = 1, shuffle: bool = False):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._cursor = 0
        self._order = np.arange(X.shape[0])

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples() / self.batch_size))

    def num_samples(self) -> int:
        return self.X.shape[0]

    def __iter__(self):
        self._cursor = 0
        n = self.num_samples()
        self._order = np.random.permutation(n) if self.shuffle else np.arange(n)
        return self

    def __next__(self):
        if self._cursor >= self.num_samples():
            raise StopIteration
        lo = self._cursor
        hi = min(lo + self.batch_size, self.num_samples())
        self._cursor = hi
        batch = self._order[lo:hi]
        return self.X[batch], self.y[batch]