import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class AdversarialAgent:
    def __init__(self, num_users: int, gamma: float):
        self.log = logging.getLogger(f"\033[94m[{self.__class__.__name__}]\033[0m")
        self.num_users = num_users
        self.gamma = gamma
        self.log.info("Users: %d, Gamma: %f.", self.num_users, self.gamma)
    
    def get_input(self, random=False):
        """Generate custom adversarial input"""
        if random:
            return np.random.rand(self.num_users)
        else:
            noise = np.random.uniform(low=0.0, high=self.gamma, size=(self.num_users, 1))
            attacker_input = (np.ones((self.num_users, 1)) - np.ones((self.num_users, 1)) * self.gamma) - noise
            return attacker_input.flatten()