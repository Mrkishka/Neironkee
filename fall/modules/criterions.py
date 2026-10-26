import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """Среднеквадратичная ошибка."""

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert input.shape == target.shape, 'shape mismatch'
        diff = input - target
        return np.mean(diff * diff)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert input.shape == target.shape, 'shape mismatch'
        return 2.0 * (input - target) / input.size


class CrossEntropyLoss(Criterion):
    """Cross-entropy по логитам, с опциональным label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def _smoothed_target(self, target: np.ndarray, num_classes: int) -> np.ndarray:
        dist = np.full((target.size, num_classes),
                       self.label_smoothing / num_classes)
        dist[np.arange(target.size), target] += 1.0 - self.label_smoothing
        return dist

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        log_probs = self.log_softmax.forward(input)
        b, c = input.shape
        if self.label_smoothing > 0.0:
            dist = self._smoothed_target(target, c)
            loss = -(dist * log_probs).sum() / b
        else:
            loss = -log_probs[np.arange(b), target].mean()
        return np.array(loss, dtype=input.dtype)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        probs = np.exp(self.log_softmax.forward(input))
        b, c = input.shape
        if self.label_smoothing > 0.0:
            dist = self._smoothed_target(target, c)
            return (probs - dist) / b
        grad = probs.copy()
        grad[np.arange(b), target] -= 1.0
        return grad / b