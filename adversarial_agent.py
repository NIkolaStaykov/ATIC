import numpy as np


class AdversarialAgent:
    
    def __init__(self, num_users: int, gamma: float):

        self.num_users = num_users
        self.gamma = gamma
        print(f"\033[94m[AdversarialAgent]\033[0m Users: {self.num_users}, Gamma: {self.gamma}.")
        
    def get_random_input(self):
        """Generate random input for the adversarial agent"""
        return np.random.rand(self.num_users)
    
    def attack(self):
        """Generate custom adversarial input"""
        noise = np.random.uniform(low=0.0, high=self.gamma, size=(self.num_users, 1))
        attacker_input = (np.ones((self.num_users, 1)) - np.ones((self.num_users, 1)) * self.gamma) - noise
        return attacker_input.flatten()