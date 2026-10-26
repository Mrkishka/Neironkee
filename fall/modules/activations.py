import numpy as np
from scipy.special import erf, logsumexp, expit
from .base import Module


class ReLU(Module):
    """Поэлементный ReLU."""
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(input, 0.0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (input > 0.0)


class Sigmoid(Module):
    """Поэлементная сигмоида."""
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = expit(input)
        return grad_output * s * (1.0 - s)


class GELU(Module):
    """y = x * Phi(x), Phi — функция распределения N(0, 1)."""
    _INV_SQRT2 = 1.0 / np.sqrt(2.0)
    _INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return 0.5 * input * (1.0 + erf(input * self._INV_SQRT2))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        cdf = 0.5 * (1.0 + erf(input * self._INV_SQRT2))
        pdf = self._INV_SQRT_2PI * np.exp(-0.5 * input * input)
        return grad_output * (cdf + input * pdf)


class Softmax(Module):
    """Softmax по последней оси."""
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        exps = np.exp(input - input.max(axis=-1, keepdims=True))
        return exps / exps.sum(axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = self.compute_output(input)
        return s * (grad_output - np.sum(grad_output * s, axis=-1, keepdims=True))


class LogSoftmax(Module):
    """LogSoftmax по последней оси."""
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return input - logsumexp(input, axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        probs = np.exp(self.compute_output(input))
        return grad_output - np.sum(grad_output, axis=-1, keepdims=True) * probs