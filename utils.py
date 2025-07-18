import numpy as np

class NoiseGenerator:
    def __init__(self, noise_level: float):
        self.noise_level = noise_level

    def generate(self, shape: tuple) -> np.ndarray:
        return np.random.normal(0, self.noise_level, shape)