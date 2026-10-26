import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """SGD с momentum / weight_decay / nesterov."""

    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def step(self):
        params = self.module.parameters()
        grads = self.module.parameters_grad()
        if 'velocity' not in self.state:
            self.state['velocity'] = [np.zeros_like(p) for p in params]

        for p, g, v in zip(params, grads, self.state['velocity']):
            update = g
            if self.weight_decay != 0.0:
                update = update + self.weight_decay * p
            if self.momentum != 0.0:
                v *= self.momentum
                v += update
                update = update + self.momentum * v if self.nesterov else v
            p -= self.lr * update


class Adam(Optimizer):
    """Adam."""

    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(module)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        params = self.module.parameters()
        grads = self.module.parameters_grad()
        if 'state' not in self.state:
            self.state['state'] = {
                'm': [np.zeros_like(p) for p in params],
                'v': [np.zeros_like(p) for p in params],
                't': 0,
            }
        s = self.state['state']
        s['t'] += 1
        t = s['t']
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t

        for p, g, m, v in zip(params, grads, s['m'], s['v']):
            d = g if self.weight_decay == 0.0 else g + self.weight_decay * p
            m *= self.beta1
            m += (1.0 - self.beta1) * d
            v *= self.beta2
            v += (1.0 - self.beta2) * d * d
            p -= (self.lr / bc1) * m / (np.sqrt(v / bc2) + self.eps)