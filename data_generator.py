from dataclasses import dataclass
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

from adversarial_agent import AdversarialAgent
from controller import Controller
from controller import SensitivityEstimator

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


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

    def get_true_sensitivity_matrix(self):
        """Compute true sensitivity matrix for comparison"""
        N = self.user_influence_matrix.shape[0]
        user_weights = self.get_user_influence_weights()
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
            num_users=self.num_users,
            control_gain=config['controller']['control_gain'],
        )

        # initialize sensitivity estimator using pykalman
        self.sensitivity_estimator = SensitivityEstimator(
            n_users=self.num_users,
            process_noise_var=config['kalman_filter']['process_noise_var'],
            measurement_noise_var=config['kalman_filter']['measurement_noise_var'],
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
        
    def step(self, control_input: np.ndarray, attacker_input: np.ndarray):
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
            
            control_input = self.controller.recommend()
            
            if step < 10: 
                # Exploration phase for Kalman Filter
                attacker_input = self.adversary.get_random_input()
            else:
                attacker_input = self.adversary.attack()

            self.step(control_input, attacker_input)

            # Sensitivity estimation using pykalman
            sensitivity_estimate = None
            kalman_covariance_trace = None
            estimation_error = None
            
            if step > 0 and prev_control_input is not None:
                # Compute deltas for Kalman filter
                delta_x_ss = self.state.opinion_state - prev_opinion_state
                delta_p = control_input - prev_control_input
                
                # Update sensitivity estimator
                self.sensitivity_estimator.update(delta_x_ss, delta_p)
                
                # Get current estimates
                sensitivity_estimate = self.sensitivity_estimator.get_sensitivity_matrix()
                kalman_covariance_trace = self.sensitivity_estimator.get_covariance_trace()
                
                # Compute estimation error
                true_sensitivity = self.state.get_true_sensitivity_matrix()
                estimation_error = np.linalg.norm(sensitivity_estimate - true_sensitivity, 'fro')
                
                # Store for analysis
                self.sensitivity_estimator.estimation_errors.append(estimation_error)
                self.sensitivity_estimator.error_steps.append(step)
                
                # self.log.info progress every 10 steps (for debugging, can be removed)
                if self.debug and step % 10 == 0:
                    self.log.debug("\nStep %d:", step)
                    self.log.debug("Estimated sensitivity:\n%s", np.array2string(sensitivity_estimate, formatter={'float_kind':'{:0.2f}'.format}))
                    self.log.debug("True sensitivity:\n%s", np.array2string(true_sensitivity, formatter={'float_kind':'{:0.2f}'.format}))
                    self.log.debug("Frobenius norm of estimation error: %.6f", estimation_error)
                    self.log.debug("Covariance trace (uncertainty): %.6f", kalman_covariance_trace)
                    self.log.debug("-" * 60)
            
            # Store current values for next iteration
            prev_control_input = control_input.copy()
            prev_opinion_state = self.state.opinion_state.copy()

            yield {
                'step': step,
                'opinion_state': self.state.opinion_state.copy(),
                'control_input': control_input.copy(),
                'attacker_input': attacker_input.copy(),
                'sensitivity_estimate': sensitivity_estimate,
                'kalman_covariance_trace': kalman_covariance_trace,
                'estimation_error': estimation_error
            }
    

    def plot_estimation_error(self, filename: str = "pykalman_estimation_error.png"):
        """Plot the estimation error and other metrics"""

        errors = self.sensitivity_estimator.estimation_errors
        steps = self.sensitivity_estimator.error_steps
        
        if not errors:
            self.log.info("No estimation errors to plot.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Linear scale
        ax1.plot(steps, errors, 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Frobenius Norm Error')
        ax1.set_title('Sensitivity Estimation Error (Linear)')
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.semilogy(steps, errors, 'r-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Frobenius Norm Error (log)')
        ax2.set_title('Sensitivity Estimation Error (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Moving average
        if len(errors) > 5:
            window = min(10, len(errors) // 3)
            moving_avg = np.convolve(errors, np.ones(window)/window, mode='valid')
            moving_steps = steps[window-1:]
            
            ax3.plot(steps, errors, 'b-', alpha=0.3, label='Raw')
            ax3.plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'MA(window={window})')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Frobenius Norm Error')
            ax3.set_title('Smoothed Estimation Error')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Statistics
        ax4.axis('off')
        stats_text = f"""
        Filter Statistics:
        
        Final Error: {errors[-1]:.6f}
        Mean Error: {np.mean(errors):.6f}
        Min Error: {np.min(errors):.6f}
        Max Error: {np.max(errors):.6f}
        Std Error: {np.std(errors):.6f}
        
        Total Updates: {len(errors)}
        Users: {self.num_users}
        State Dimension: {self.num_users**2}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info("Plot saved as %s", filename)
