import numpy as np


class DifficultyLevel:
    def __init__(self):
        self.level = 1  # 1..5
        self.history = []

    def push(self, correct: int):
        self.history.append(int(bool(correct)))
        if len(self.history) > 40:
            self.history.pop(0)

    def rolling_acc(self, k: int = 10) -> float:
        if not self.history:
            return 1.0
        tail = self.history[-k:]
        return float(np.mean(tail)) if tail else 1.0

    def tune(self):
        acc = self.rolling_acc(12)
        if acc < 0.6 and self.level > 1:
            self.level -= 1
        elif acc > 0.85 and self.level < 5:
            self.level += 1

    def noise_params(self):
        return {
            1: (2, 1),
            2: (3, 1),
            3: (3, 2),
            4: (4, 2),
            5: (5, 2),
        }[self.level]

    def rotation_set(self):
        return {
            1: [0],
            2: [0, 90],
            3: [0, 90, 180],
            4: [0, 90, 180, 270],
            5: [0, 45, 90, 180, 270],
        }[self.level]

    def occlusion_range(self):
        return {
            1: (0.0, 0.0),
            2: (0.12, 0.20),
            3: (0.15, 0.28),
            4: (0.18, 0.35),
            5: (0.22, 0.45),
        }[self.level]
