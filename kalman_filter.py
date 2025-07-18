import numpy as np


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
        
        print(f"\033[96m[SensitivityEstimator]\033[0m Users: {self.n_users}, Process Noise Variance: {process_noise_var}, Measurement Noise Variance: {measurement_noise_var}.")
    
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