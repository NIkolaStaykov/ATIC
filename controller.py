import numpy as np


class Controller:
    
    def __init__(self, num_users: int, control_gain: float):
        self.num_users = num_users
        self.control_gain = control_gain
        print(f"\033[95m[Controller]\033[0m Users: {self.num_users}, control gain={self.control_gain}.")
    
    def recommend(self):
        control_inputs = np.random.rand(self.num_users)
        return control_inputs