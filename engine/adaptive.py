import numpy as np
from scipy.special import expit
from models import Item


class AdaptiveEngine:
    def __init__(self, init_theta: float = 0.0, lr: float = 0.25):
        self.theta = init_theta
        self.lr = lr

    @staticmethod
    def prob_correct(theta: float, a: float, b: float) -> float:
        return expit(a * (theta - b))

    def expected_information(self, a: float, b: float) -> float:
        p = self.prob_correct(self.theta, a, b)
        return a * a * p * (1 - p)

    def select_item(self, candidates):
        return max(candidates, key=lambda it: self.expected_information(it.a, it.b))

    def update(self, item: Item, response: int):
        p = self.prob_correct(self.theta, item.a, item.b)
        grad = (response - p) * item.a
        self.theta = float(np.clip(self.theta + self.lr * grad, -4.0, 4.0))
