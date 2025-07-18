import numpy as np
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class SensitivityEstimator:
    """Kalman Filter implementation"""
    
    def __init__(self, cfg, n_users: int):
        self.log = logging.getLogger(f"\033[96m[{self.__class__.__name__}]\033[0m")
        self.n_users = n_users
        self.n_states = n_users ** 2  # Size of vectorized sensitivity matrix
        
        # Parameters for Kalman updates
        self.F = np.eye(self.n_states)  # Transition Matrix
        self.Q = np.eye(self.n_states) * cfg["process_noise_var"]  # Process noise covariance
        self.R = np.eye(n_users) * cfg["measurement_noise_var"]    # Measurement noise covariance
        
        # Initialize filter state
        self.state_mean = np.zeros(self.n_states)
        self.state_covariance = np.eye(self.n_states) * 1.0
        
        # History for tracking
        self.estimation_errors = []
        self.error_steps = []
        self.sensitivity_estimates = []
        
        self.log.info("Users: %d, Process Noise Variance: %.4f, Measurement Noise Variance: %.4f.", self.n_users, process_noise_var, measurement_noise_var)
    
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

class Controller:
    # NOTE: There is deffinitely a better way to pass repeating values with hydra
    def __init__(self, cfg, num_users):
        self.log = logging.getLogger(f"\033[96m[{self.__class__.__name__}]\033[0m")

        self.num_users = num_users
        self.control_gain = cfg["control_gain"]
        self.sensitivity_estimator = SensitivityEstimator(cfg = cfg["kalman_filter"], n_users = self.num_users)

        self.log.info("Users: %d, control gain=%f.", self.num_users, self.control_gain)
    
    def recommend(self):
        control_inputs = np.random.rand(self.num_users)
        return control_inputs
    
    def step(self, state):
        # Compute deltas for Kalman filter
        delta_x_ss = state.opinion_state - prev_opinion_state
        delta_p = control_input - self.prev_control_input
        
        # Update sensitivity estimator
        self.sensitivity_estimator.update(delta_x_ss, delta_p)
        
        # Get current estimates
        sensitivity_estimate = self.sensitivity_estimator.get_sensitivity_matrix()
        kalman_covariance_trace = self.sensitivity_estimator.get_covariance_trace()

        # Store current values for next iteration
        prev_control_input = control_input.copy()
        prev_opinion_state = self.state.opinion_state.copy()