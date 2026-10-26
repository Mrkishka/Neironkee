import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """Аффинное преобразование: y = x @ W.T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / np.sqrt(in_features)
        self.weight = np.random.uniform(-bound, bound, (out_features, in_features))
        self.bias = np.random.uniform(-bound, bound, out_features) if bias else None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        out = input @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0.0)
        if self.bias is not None:
            self.grad_bias.fill(0.0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.bias is not None else [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.bias is not None else [self.grad_weight]

    def __repr__(self) -> str:
        return (f'Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')


class BatchNormalization(Module):
    """Batch normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        if affine:
            self.weight = np.ones(num_features)
            self.bias = np.zeros(num_features)
            self.grad_weight = np.zeros(num_features)
            self.grad_bias = np.zeros(num_features)
        else:
            self.weight = self.bias = None
            self.grad_weight = self.grad_bias = None

        # кэш для обратного прохода
        self._x_hat = None
        self._inv_std = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            batch_mean = input.mean(axis=0)
            centered = input - batch_mean
            var = (centered ** 2).mean(axis=0)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = centered * inv_std

            n = input.shape[0]
            unbiased = var * n / (n - 1) if n > 1 else var
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * unbiased

            self._x_hat = x_hat
            self._inv_std = inv_std
        else:
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            self._x_hat = (input - self.running_mean) * inv_std
            self._inv_std = inv_std

        out = self._x_hat
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad_hat = grad_output * self.weight if self.affine else grad_output
        if self.training:
            mean_g = grad_hat.mean(axis=0)
            mean_gx = (grad_hat * self._x_hat).mean(axis=0)
            return self._inv_std * (grad_hat - mean_g - self._x_hat * mean_gx)
        return grad_hat * self._inv_std

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        if self.affine:
            self.grad_weight += (grad_output * self._x_hat).sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0.0)
            self.grad_bias.fill(0.0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return (f'BatchNormalization(num_features={self.running_mean.size}, '
                f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})')


class Dropout(Module):
    """Inverted dropout."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
        self._mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return input
        keep = 1.0 - self.p
        self._mask = (np.random.random(input.shape) < keep) / keep
        return input * self._mask

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_output
        return grad_output * self._mask

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """Последовательный контейнер модулей."""

    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        x = input
        for layer in self.modules:
            x = layer.forward(x)
        return x

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for idx in range(len(self.modules) - 1, -1, -1):
            layer = self.modules[idx]
            layer_in = input if idx == 0 else self.modules[idx - 1].output
            grad = layer.backward(layer_in, grad)
        return grad

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for layer in self.modules:
            layer.train()

    def eval(self):
        for layer in self.modules:
            layer.eval()

    def zero_grad(self):
        for layer in self.modules:
            layer.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [p for layer in self.modules for p in layer.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [g for layer in self.modules for g in layer.parameters_grad()]

    def __repr__(self) -> str:
        body = '\n'.join('    ' + repr(layer) for layer in self.modules)
        return f'Sequential(\n{body}\n)'