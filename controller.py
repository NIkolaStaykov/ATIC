import numpy as np
import collections
from statistics import mean
import logging
import sys
from enum import Enum
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

ControllerType = Enum('ControllerType', 'RANDOM ADVERSARIAL')

class SensitivityEstimator:
    """Kalman Filter implementation"""
    
    def __init__(self, cfg, n_users: int):
        self.log = logging.getLogger(f"\033[96m{self.__class__.__name__}\033[0m")
        self.n_users = n_users
        self.n_states = n_users ** 2  # Size of vectorized sensitivity matrix
        
        # Parameters for Kalman updates
        self.F = np.eye(self.n_states)  # Transition Matrix
        self.Q = np.eye(self.n_states) * cfg["process_noise_var"]  # Process noise covariance
        self.R = np.eye(n_users) * cfg["measurement_noise_var"]    # Measurement noise covariance
        self.T = cfg['trigger_period'] # Trigger period for Kalman Filter posterior update
        self.num_steps_between_random = cfg['num_steps_between_random'] # Period for Persistent Excitation of the system
        # Initialize filter state
        self.state_mean = np.zeros(self.n_states)
        self.state_covariance = np.eye(self.n_states) * 1.0
        
        # History for tracking
        self.estimation_errors = []
        self.error_steps = []
        self.sensitivity_estimates = []
        
        self.log.info("Users: %d, Process Noise Variance: %.4f, Measurement Noise Variance: %.4f.", self.n_users, cfg["process_noise_var"], cfg["measurement_noise_var"])
    
    def update(self, delta_x_ss: np.ndarray, delta_p: np.ndarray, step):
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
        
        # Triggered posterior update every T steps
        if (step % self.T == 0):
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
        self.log = logging.getLogger(f"\033[96m{self.__class__.__name__}\033[0m")
        self.log.setLevel(cfg["log_level"].upper())

        self.controller_type: ControllerType = ControllerType[cfg["type"].upper()]
        self.log.info("Controller type: %s", self.controller_type.name)

        self.num_users = num_users
        self.state_history_length = cfg["state_history_length"]
        self.warmup = True
        self.warmup_len = cfg["warmup_len"]
        self.control_gain = cfg["control_gain"]
        self.sensitivity_estimator = SensitivityEstimator(cfg = cfg["kalman_filter"], n_users = self.num_users)

        self.control_input = np.zeros(self.num_users)
        self.prev_control_inputs = np.zeros((2, self.num_users))  # Store previous control inputs for Kalman update
        self.last_non_random_control_input = np.zeros(self.num_users)
        self.prev_opinion_states = np.zeros((self.state_history_length, self.num_users))  # Initialize with zeros

        self.kalman_covariance_trace = None
        self.sensitivity_estimate = np.zeros([self.num_users, self.num_users])

        self.log.info("Users: %d, control gain=%f.", self.num_users, self.control_gain)
        
        self._target_opinion_states = np.zeros(self.num_users)  # Target opinion states for the controller

    @property
    def target_opinion_states(self):
        return self._target_opinion_states
    
    @target_opinion_states.setter
    def target_opinion_states(self, value):
        if len(value) != self.num_users:
            raise ValueError(f"Target opinion states must have length {self.num_users}, got {len(value)}")
        self._target_opinion_states = value
    
    def get_input(self):
        return self.control_input
    
    def is_converged(self):
        """Check if the opinions have settled"""
        if len(self.prev_opinion_states) < self.state_history_length:
            return False
        
        # Check if the moving average settled
        moving_avg_old = np.mean(self.prev_opinion_states[1:, :], axis=0)
        moving_avg_new = np.mean(self.prev_opinion_states[:-1, :], axis=0)
        return np.all(np.abs(moving_avg_old - moving_avg_new) < 1e+4)
    
    def time_for_random_step(self, step):
        """Check if it's time for a random step"""
        return (step % self.sensitivity_estimator.num_steps_between_random == 0)

        
    def step(self, state, step):
        # Update Kalman filter if we have previous data
        if step > 0:
            delta_x_ss = state.opinion_state - self.prev_opinion_states[-1]
            delta_p = self.prev_control_inputs[0, :] - self.prev_control_inputs[1, :]
            # self.log.info(f"delta_x_ss: {delta_x_ss}, delta_p: {delta_p}")
            if np.linalg.norm(delta_p) > 1e-6:
                self.sensitivity_estimator.update(delta_x_ss, delta_p, step)
                
                self.sensitivity_estimate = self.sensitivity_estimator.get_sensitivity_matrix()
                self.kalman_covariance_trace = self.sensitivity_estimator.get_covariance_trace()

        if self.warmup:
            self.log.debug("Warmup, stay cozy")
            # If no previous control input, initialize to zero
            self.control_input = np.random.uniform(low=-0.1, high=0.1, size=self.num_users)
            if step == self.warmup_len: self.warmup = False

        elif self.time_for_random_step(step) and step > self.warmup_len:
            self.control_input = np.random.uniform(low=-1.0, high=1.0, size=self.num_users)

        elif self.is_converged():
            # print("Step %d: Opinions have converged, generating new controller input." % step)
            # Generate new control input
            if self.controller_type == ControllerType.ADVERSARIAL:
                                
                M_matrix = state.get_true_sensitivity_matrix()
                # M_matrix = self.sensitivity_estimate
                    
                # (dx_i / dp_i)(x_i - x^T_i)   %%% (dx_i / dp_i) = M_ii
                # gain * (dx_i / dp_i)(x_i - x^T_i)
                # M_diag = diag(diag(M))

                diagonal_M = np.diag(np.diag(M_matrix))
                control_input_unclipped = self.prev_control_inputs[0, :] - \
                    self.control_gain * diagonal_M @ (self.prev_opinion_states[0] - self._target_opinion_states)
                    
                # Clip control input to [-1, 1]
                # self.control_input = control_input_unclipped / max(control_input_unclipped)
                self.control_input = np.clip(control_input_unclipped, -1.0, 1.0)
            elif self.controller_type == ControllerType.RANDOM:
                self.control_input = np.random.uniform(low=-1.0, high=1.0, size=self.num_users)
            else:
                raise ValueError(f"Unknown ControllerType: {self.controller_type}")
            
            self.last_non_random_control_input = self.control_input.copy()
        else:
            self.control_input = self.last_non_random_control_input.copy()


        # Store previous control input before generating new one
        # Store current values for next iteration
        self.prev_control_inputs[1, :] = self.prev_control_inputs[0, :].copy()
        self.prev_control_inputs[0, :] = self.control_input.copy()
        self.prev_opinion_states = np.roll(self.prev_opinion_states, 1, axis=0)
        self.prev_opinion_states[0, :] = state.opinion_state.copy()
        

