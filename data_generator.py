from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    # Model parameters
    user_influence_matrix: np.ndarray
    controller_influences: np.ndarray
    attacker_influences: np.ndarray
    # State
    opinion_state: np.ndarray

    def get_user_influence_weights(self):
        return np.eye(self.user_influence_matrix.shape[0]) - self.attacker_influences - self.controller_influences

class NoiseGenerator:
    def __init__(self, noise_level: float):
        self.noise_level = noise_level

    def generate(self, shape: tuple) -> np.ndarray:
        return np.random.normal(0, self.noise_level, shape)

class DataGenerator:
    def __init__(self, config):
        self.num_users = config['num_users']
        self.state = self.generate_initial_state(config)

    @staticmethod
    def generate_initial_state(config):
        # Simulate the generation of an initial state based on the configuration
        user_influence_matrix = np.random.rand(config['num_users'], config['num_users'])
        controller_influences = np.diag(np.random.rand(config['num_users']))
        attacker_influences = np.diag(np.random.rand(config['num_users']))
        initial_opinions = np.random.rand(config['num_users'])

        return State(
            user_influence_matrix=user_influence_matrix,
            controller_influences=controller_influences,
            attacker_influences=attacker_influences,
            opinion_state=initial_opinions
        )
    def generate_attacker_input(self, N, gamma: float = 0.1):
        noise = np.random.uniform(low=0.0, high=gamma, size=(N,1))
        attacker_input = (np.ones((N,1))-np.ones((N,1))*gamma) - noise
        return attacker_input


    def step(self, control_input: np.ndarray, attacker_input: np.ndarray):
        # Extended Friedkin Johnsen model step
        user_weights = self.state.get_user_influence_weights()

        self.state.opinion_state = (
            user_weights @ self.state.user_influence_matrix @ self.state.opinion_state +
            self.state.controller_influences @ control_input +
            self.state.attacker_influences @ attacker_input
        )

    def generate(self):
        for step in range(100):
            control_input = np.random.rand(self.num_users)
            attacker_input = self.generate_attacker_input(self.num_users)

            self.step(control_input, attacker_input)

            yield {
                'step': step,
                'opinion_state': self.state.opinion_state.copy(),
                'control_input': control_input.copy(),
                'attacker_input': attacker_input.copy()
            }