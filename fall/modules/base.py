import numpy as np
from abc import ABC, abstractmethod
from typing import List


class Module(ABC):

    def __init__(self):
        self.output = None
        self.training = True

    # --- обязательный контракт ---
    @abstractmethod
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """Прямой проход: y = f(x)."""
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Градиент по входу: dl/dx = dl/df * df/dx."""
        raise NotImplementedError

    # --- опциональные хуки ---
    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """Накопить градиенты параметров."""
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass

    def parameters(self) -> List[np.ndarray]:
        return []

    def parameters_grad(self) -> List[np.ndarray]:
        return []

    # --- основной API ---
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.output = self.compute_output(input)
        return self.output

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad_input = self.compute_grad_input(input, grad_output)
        self.update_grad_parameters(input, grad_output)
        return grad_input

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Criterion(ABC):
    """Базовый класс для функций потерь."""

    def __init__(self):
        self.output = None

    @abstractmethod
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self.output = self.compute_output(input, target)
        return self.output

    def backward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self.compute_grad_input(input, target)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Optimizer(ABC):
    """Базовый класс оптимизаторов."""

    def __init__(self, module: Module):
        self.module = module
        self.state = {}

    def zero_grad(self):
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        raise NotImplementedError