from dataclasses import dataclass
import logging
import sys
import numpy as np
from enum import Enum

from adversarial_agent import AdversarialAgent
from controller import Controller
from controller import SensitivityEstimator

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

DataGeneratorType = Enum('DataGeneratorType', 'PURE WITH_ACTORS ONLY_BAD')


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

    def get_user_influence_weights(self, type: DataGeneratorType):
        """Return correct influence weights depending on specified scenario type"""
        if type == DataGeneratorType.PURE:
            return np.eye(self.user_influence_matrix.shape[0])
        elif type == DataGeneratorType.ONLY_BAD:
            return np.eye(self.user_influence_matrix.shape[0]) - self.attacker_influences
        elif type == DataGeneratorType.WITH_ACTORS:
            return np.eye(self.user_influence_matrix.shape[0]) - self.attacker_influences - self.controller_influences
        else:
            raise ValueError(f"Unknown DataGeneratorType: {type}")

    def get_true_sensitivity_matrix(self):
        """Compute true sensitivity matrix for comparison"""
        N = self.user_influence_matrix.shape[0]
        user_weights = self.get_user_influence_weights(DataGeneratorType.WITH_ACTORS)
        A_tilde = user_weights @ self.user_influence_matrix
        Gamma_p = self.controller_influences
        
        try:
            sensitivity_matrix = np.linalg.inv(np.eye(N) - A_tilde) @ Gamma_p
        except np.linalg.LinAlgError:
            sensitivity_matrix = np.linalg.pinv(np.eye(N) - A_tilde) @ Gamma_p

        return sensitivity_matrix


class DataGenerator:
    
    def __init__(self, config):
        np.random.seed(config['seed'])
        self.type = DataGeneratorType[config['type'].upper()]
        self.debug = config['debug']
        self.log = logging.getLogger(f"\033[96m{self.__class__.__name__}\033[0m")
        if self.debug: self.log.setLevel(level = logging.DEBUG)

        # self.log.info("DataGenerator type: %s, Debug mode: %s.", self.type.name, self.debug)
        # initialize network
        self.num_users = config['network']['num_users']
        self.num_steps = config['network']['num_steps']
        # self.log.info(f"Users: {self.num_users}, Steps: {self.num_steps}.")
        self.state = self.generate_initial_state(config)

        # Get time durations of each phase
        self.T_pure = config['T_pure'] # Duration of pure phase
        self.T_only_bad = config['T_only_bad'] # Duration of phase with adversary
        self.T_with_actors = config['T_with_actors'] # Duration of phase with controller
        
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
        
    def generate_initial_state(self, config):
        """Simulate the generation of an initial state based on the configuration"""
        n_users = config['network']['num_users']
        user_influence_matrix = np.random.rand(n_users, n_users)
        # Make the matrix row-stochastic (rows sum to 1)
        user_influence_matrix = user_influence_matrix / user_influence_matrix.sum(axis=1, keepdims=True)
        
        if self.type == DataGeneratorType.PURE:
            controller_influences = np.zeros((n_users, n_users))
            attacker_influences = np.zeros((n_users, n_users))
        elif self.type == DataGeneratorType.WITH_ACTORS:
            # If actors are present, generate a random influence vector
            # This could be modified to include specific actor influences
            total_gamma_vec = np.random.rand(n_users)
            controller_influences = np.diag(np.random.uniform(low=np.zeros(n_users), high=total_gamma_vec))
            attacker_influences = np.diag(total_gamma_vec) - controller_influences
        elif self.type == DataGeneratorType.ONLY_BAD:
            # If only bad actors are present, generate a random influence vector
            total_gamma_vec = np.random.rand(n_users)
            controller_influences = np.zeros((n_users, n_users))
            attacker_influences = np.diag(total_gamma_vec)
        else:
            raise ValueError(f"Unknown DataGeneratorType: {self.type}")
        
        # self.log.info("User influence matrix:\n%s", user_influence_matrix)
        # self.log.info("Controller influences:\n%s", controller_influences)
        # self.log.info("Attacker influences:\n%s", attacker_influences)
        initial_opinions = np.random.uniform(low=-1.0, high=1.0, size=self.num_users)

        return State(
            user_influence_matrix=user_influence_matrix,
            controller_influences=controller_influences,
            attacker_influences=attacker_influences,
            opinion_state=initial_opinions
        )
        
    def update_state(self, control_input: np.ndarray, attacker_input: np.ndarray, type: DataGeneratorType):
        """Opinion dynamics step
        type = "pure": pure system dynamics with no adversary, no controller
        type = "only_bad": system dynamics with adversary, no controller
        type = "with_actors": system dynamics with adversary and actor
        """
        # Get correct user weights depending on selected scenario
        user_weights = self.state.get_user_influence_weights(type)

        # Take one step according to selected scenario
        if type == DataGeneratorType.PURE:
            self.state.opinion_state = (
                user_weights @ self.state.user_influence_matrix @ self.state.opinion_state 
            )
        elif type == DataGeneratorType.ONLY_BAD:
            self.state.opinion_state = (
                user_weights @ self.state.user_influence_matrix @ self.state.opinion_state +
                self.state.attacker_influences @ attacker_input
            )
        elif type == DataGeneratorType.WITH_ACTORS:    
            self.state.opinion_state = (
                user_weights @ self.state.user_influence_matrix @ self.state.opinion_state +
                self.state.controller_influences @ control_input +
                self.state.attacker_influences @ attacker_input
            )
        else:
            raise ValueError(f"Unknown DataGeneratorType: {type}")

    def generate(self):
        """Generate simulation steps and return relevant metrics."""

        stage: DataGeneratorType
        for step in range(self.num_steps):
            if (step < self.T_pure):
                stage = DataGeneratorType.PURE
                control_input = np.zeros(self.num_users) # placeholder controller input
                attacker_input = np.zeros(self.num_users) # placeholder attacker input

                # Update state
                self.update_state(control_input, attacker_input, stage)

                # Define placeholder estimation_error
                true_sensitivity = self.state.get_true_sensitivity_matrix()
                estimation_error = 0
            
            if (step >= self.T_pure and step < self.T_only_bad + self.T_pure):   
                stage = DataGeneratorType.ONLY_BAD
                control_input = np.zeros(self.num_users) # placeholder input                             
                attacker_input = self.adversary.get_input() # get attacker input

                # Update state
                self.update_state(control_input, attacker_input, stage)
                
                # Define placeholder estimation_error
                true_sensitivity = self.state.get_true_sensitivity_matrix()
                estimation_error = 0

            if (step >= self.T_only_bad + self.T_pure):
                stage = DataGeneratorType.WITH_ACTORS
                # Update Kalman filter in controller, then get control input
                self.controller.step(self.state, step=step-self.T_only_bad-self.T_pure)   # hardcoded step offset
                control_input = self.controller.get_input()
            
                # Get attacker input
                attacker_input = self.adversary.get_input()

                # Update state
                self.update_state(control_input, attacker_input, stage)
            
                # Compute estimation error
                true_sensitivity = self.state.get_true_sensitivity_matrix()
                estimation_error = np.linalg.norm(self.controller.sensitivity_estimate - true_sensitivity, 'fro')
            
            # self.log.info progress every 10 steps (for debugging, can be removed)
            if step % 10 == 0:
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
                'estimation_error': estimation_error,
                "stage": stage.name
            }
    