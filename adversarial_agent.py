import numpy as np
import logging
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class AdversarialAgent:
    def __init__(self, num_users: int, gamma: float, initial_opinions: Optional[np.ndarray] = None):
        self.log = logging.getLogger(f"\033[96m{self.__class__.__name__}\033[0m")
        self.num_users = num_users
        self.gamma = gamma
        self.push_directions = np.sign(initial_opinions) if initial_opinions is not None else np.ones(num_users)
        self.log.info("Users: %d, Gamma: %f.", self.num_users, self.gamma)
    
    def get_input(self, random=False):
        """Generate custom adversarial input"""
        if random:
            return np.random.uniform(low=-1.0, high=1.0, size=self.num_users)
        else:
            noise = np.random.uniform(low=0.0, high=self.gamma, size=(self.num_users, 1))
            attacker_input = (np.ones((self.num_users, 1)) - np.ones((self.num_users, 1)) * self.gamma) + noise
            self.log.debug("Adversarial input: %s", attacker_input.flatten())
            self.log.debug("Push directions: %s", self.push_directions)
            self.log.debug("Attacker input: %s", attacker_input.flatten() * self.push_directions)
            return attacker_input.flatten()