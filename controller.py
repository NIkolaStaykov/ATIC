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
        
        self.log.info("Users: %d, Process Noise Variance: %.4f, Measurement Noise Variance: %.4f.", self.n_users, cfg["process_noise_var"], cfg["measurement_noise_var"])
    
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
        # FIXME: try except used for flow control
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

        self.control_input = np.zeros(self.num_users)
        self.prev_control_input = None
        self.prev_opinion_state = None

        self.kalman_covariance_trace = None
        self.sensitivity_estimate = np.zeros([self.num_users, self.num_users])

        self.log.info("Users: %d, control gain=%f.", self.num_users, self.control_gain)
    
    def get_input(self):
        return self.control_input
    
    def step(self, state):
        # TODO: Review! Right now the first two steps give control inputs 0 and random respectively and then the actual optimization starts
        # The Kalman filter is updated after the controller input is calculated
        # Possibly we also need a *predict* f-n in the Kalman filter class

        if self.prev_control_input is not None:
            # Compute deltas for Kalman filter
            delta_x_ss = state.opinion_state - self.prev_opinion_state
            delta_p = self.control_input - self.prev_control_input
            
            # Update sensitivity estimator
            self.sensitivity_estimator.update(delta_x_ss, delta_p)            
            # Get current estimates, to be used in the gradient descent step
            self.sensitivity_estimate = self.sensitivity_estimator.get_sensitivity_matrix()
            self.kalman_covariance_trace = self.sensitivity_estimator.get_covariance_trace()

            # Get the next control input
            # TODO: Implement gradient descent
            self.prev_control_input = self.control_input.copy()
            self.control_input = np.random.rand(self.num_users)

        else:
            # First step, no previous values, cannot update Kalman because we need the diff of current and prev state
            # We have to use the prediction from the kalman filter 
            self.prev_control_input = self.control_input.copy()
            self.control_input = np.random.rand(self.num_users)

        self.prev_opinion_state = state.opinion_state