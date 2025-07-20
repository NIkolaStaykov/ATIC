from dataclasses import dataclass
import logging
import sys
import numpy as np

from adversarial_agent import AdversarialAgent
from controller import Controller
from controller import SensitivityEstimator

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


@dataclass
class State:
    # --- STATIC ---
    # Base model parameters
    user_influence_matrix: np.ndarray
    controller_influences: np.ndarray
    attacker_influences: np.ndarray
    # --- DYNAMIC ---
    # State
    opinion_state: np.ndarray

    def get_user_influence_weights(self):
        return np.eye(self.user_influence_matrix.shape[0]) - self.attacker_influences - self.controller_influences

    def get_true_sensitivity_matrix(self):
        """Compute true sensitivity matrix for comparison"""
        N = self.user_influence_matrix.shape[0]
        user_weights = self.get_user_influence_weights()
        A_tilde = user_weights @ self.user_influence_matrix
        Gamma_p = self.controller_influences
        # FIXME: Try except shouldn't be used for flow control
        try:
            sensitivity_matrix = np.linalg.inv(np.eye(N) - A_tilde) @ Gamma_p
        except np.linalg.LinAlgError:
            sensitivity_matrix = np.linalg.pinv(np.eye(N) - A_tilde) @ Gamma_p

        return sensitivity_matrix


class DataGenerator:
    
    def __init__(self, config):
        np.random.seed(config['seed'])
        self.debug = config['debug']
        self.log = logging.getLogger(f"\033[96m[{self.__class__.__name__}]\033[0m")
        if self.debug: self.log.setLevel(level = logging.DEBUG)

        # initialize network
        self.num_users = config['network']['num_users']
        self.num_steps = config['network']['num_steps']
        self.log.info(f"\033[93m[DataGenerator]\033[0m Users: {self.num_users}, Steps: {self.num_steps}.")
        self.state = self.generate_initial_state(config)
        
        # initialize adversarial agent
        self.adversary = AdversarialAgent(
            num_users=self.num_users, 
            gamma=config['adversary']['gamma'],
        )
        
        # initialize controller
        self.controller = Controller(
            config["controller"],
            self.num_users
        )
        
    @staticmethod
    def generate_initial_state(config):
        """Simulate the generation of an initial state based on the configuration"""
        n_users = config['network']['num_users']
        user_influence_matrix = np.random.rand(n_users, n_users)
        controller_influences = np.diag(np.random.rand(n_users))
        attacker_influences = np.diag(np.random.rand(n_users))
        initial_opinions = np.random.rand(n_users)

        return State(
            user_influence_matrix=user_influence_matrix,
            controller_influences=controller_influences,
            attacker_influences=attacker_influences,
            opinion_state=initial_opinions
        )
        
    def update_state(self, control_input: np.ndarray, attacker_input: np.ndarray):
        """Extended Friedkin Johnsen model step"""
        user_weights = self.state.get_user_influence_weights()

        self.state.opinion_state = (
            user_weights @ self.state.user_influence_matrix @ self.state.opinion_state +
            self.state.controller_influences @ control_input +
            self.state.attacker_influences @ attacker_input
        )

    def generate(self):
        """Generate simulation steps and return relevant metrics."""
        # Initialize previous values for delta computation
        prev_control_input = None
        prev_opinion_state = None

        for step in range(self.num_steps):
            
            self.controller.step(self.state)

            control_input = self.controller.get_input()
            attacker_input = self.adversary.get_input(random=(step < 10))

            # This updates the state
            self.update_state(control_input, attacker_input)

            # Compute estimation error
            true_sensitivity = self.state.get_true_sensitivity_matrix()
            estimation_error = np.linalg.norm(self.controller.sensitivity_estimate - true_sensitivity, 'fro')
            
            # self.log.info progress every 10 steps (for debugging, can be removed)
            if self.debug and step % 10 == 0:
                self.log.debug("\nStep %d:", step)
                self.log.debug("Estimated sensitivity:\n%s", np.array2string(self.controller.sensitivity_estimate, formatter={'float_kind':'{:0.2f}'.format}))
                self.log.debug("True sensitivity:\n%s", np.array2string(true_sensitivity, formatter={'float_kind':'{:0.2f}'.format}))
                self.log.debug("Frobenius norm of estimation error: %.6f", estimation_error)
                self.log.debug("Covariance trace (uncertainty): %.6f", self.controller.kalman_covariance_trace)
                self.log.debug("-" * 60)
            


            yield {
                'step': step,
                'opinion_state': self.state.opinion_state.copy(),
                'control_input': control_input.copy(),
                'attacker_input': attacker_input.copy(),
                'sensitivity_estimate': self.controller.sensitivity_estimate,
                'kalman_covariance_trace': self.controller.kalman_covariance_trace,
                'estimation_error': estimation_error
            }
    
