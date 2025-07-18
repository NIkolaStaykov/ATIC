from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

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

class NoiseGenerator:
    def __init__(self, noise_level: float):
        self.noise_level = noise_level

    def generate(self, shape: tuple) -> np.ndarray:
        return np.random.normal(0, self.noise_level, shape)

class SensitivityEstimator:
    """Kalman Filter implementation"""
    
    def __init__(self, n_users: int, process_noise_var: float = 0.01, measurement_noise_var: float = 0.1):
        self.n_users = n_users
        self.n_states = n_users ** 2  # Size of vectorized sensitivity matrix
        
        # Parameters for Kalman updates
        self.F = np.eye(self.n_states)  # Transition Matrix
        self.Q = np.eye(self.n_states) * process_noise_var  # Process noise covariance
        self.R = np.eye(n_users) * measurement_noise_var    # Measurement noise covariance
        
        # Initialize filter state
        self.state_mean = np.zeros(self.n_states)
        self.state_covariance = np.eye(self.n_states) * 1.0
        
        # History for tracking
        self.estimation_errors = []
        self.error_steps = []
        self.sensitivity_estimates = []
    
    def update(self, delta_x_ss: np.ndarray, delta_p: np.ndarray):
        """Update the Kalman filter with new measurement"""

        # Skip if delta_p is too small
        if np.linalg.norm(delta_p) < 1e-8:
            return
        
        # Prediction step: l^k = l^{k-1} + w^{k-1}
        # State prediction (random walk, mean unchanged)
        state_pred = self.F @ self.state_mean
        # Covariance prediction
        covar_pred = self.F @ self.state_covariance @ self.F.T + self.Q
        
        # Construct observation matrix H = (delta_p)^T ⊗ I_n
        H = np.kron(delta_p.reshape(1, -1), np.eye(self.n_users))
        
        # Innovation covariance
        S = H @ covar_pred @ H.T + self.R
        
        # Kalman gain
        try:
            K = covar_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = covar_pred @ H.T @ np.linalg.pinv(S)
        
        # Innovation (measurement residual)
        innovation = delta_x_ss - H @ state_pred
        
        # Update step
        self.state_mean = state_pred + K @ innovation
        self.state_covariance = (np.eye(self.n_states) - K @ H) @ covar_pred
        
        # Store the current estimate
        sensitivity_matrix = self.get_sensitivity_matrix()
        self.sensitivity_estimates.append(sensitivity_matrix.copy())
    
    def get_sensitivity_matrix(self) -> np.ndarray:
        """Convert vectorized state to sensitivity matrix"""
        return self.state_mean.reshape((self.n_users, self.n_users))
    
    def get_covariance_trace(self) -> float:
        """Get trace of state covariance as uncertainty measure"""
        return np.trace(self.state_covariance)
    
class DataGenerator:
    def __init__(self, config):
        self.num_users = config['num_users']
        self.state = self.generate_initial_state(config)

        # Initialize sensitivity estimator using pykalman
        self.sensitivity_estimator = SensitivityEstimator(
            n_users=self.num_users,
            process_noise_var=config.get('process_noise_var', 0.01),
            measurement_noise_var=config.get('measurement_noise_var', 0.1)
        )

    @staticmethod
    def generate_initial_state(config):
        """Simulate the generation of an initial state based on the configuration"""
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
        """Generate custom adversarial input"""
        noise = np.random.uniform(low=0.0, high=gamma, size=(N,1))
        attacker_input = (np.ones((N,1))-np.ones((N,1))*gamma) - noise
        return attacker_input.flatten()


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

        for step in range(100):
            control_input = np.random.rand(self.num_users)
            if step < 10: 
                # Exploration phase for Kalman Filter
                attacker_input = np.random.rand(self.num_users)
            else:
                attacker_input = self.generate_attacker_input(self.num_users)

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
                
                # Print progress every 10 steps (for debugging, can be removed)
                if step % 10 == 0:
                    print(f"\nStep {step}:")
                    print("Estimated sensitivity:\n", np.array2string(sensitivity_estimate, formatter={'float_kind':'{:0.2f}'.format}))
                    print("True sensitivity:\n", np.array2string(true_sensitivity, formatter={'float_kind':'{:0.2f}'.format}))
                    print(f"Frobenius norm of estimation error: {estimation_error:.6f}")
                    print(f"Covariance trace (uncertainty): {kalman_covariance_trace:.6f}")
                    print(f"Frobenius norm of estimation error: {estimation_error:.6f}")
                    print(f"Covariance trace (uncertainty): {kalman_covariance_trace:.6f}")
                    print("-" * 60)
            
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
            print("No estimation errors to plot.")
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
        print(f"Plot saved as {filename}")