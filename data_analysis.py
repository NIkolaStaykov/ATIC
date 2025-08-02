import matplotlib.pyplot as plt
import logging
import sys
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class Plotter:
    def __init__(self, data: pd.DataFrame, key_stats: dict):
        self.log = logging.getLogger(f"\033[96m{self.__class__.__name__}\033[0m")
        self.data = data
        self.key_stats = key_stats
        self.log_folder = HydraConfig.get().runtime.output_dir

    def plot_estimation_error(self, filename: str = "pykalman_estimation_error.png"):
        """Plot the estimation error and other metrics"""

        errors = self.data.loc[:, "estimation_error"]
        steps = self.data.loc[:, "step"]
        assert len(errors) > 0, "No estimation errors to plot"
        
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
        
        Final Error: {errors.iloc[1]:.6f} 
        Mean Error: {np.mean(errors):.6f}
        Min Error: {np.min(errors):.6f}
        Max Error: {np.max(errors):.6f}
        Std Error: {np.std(errors):.6f}
        
        Total Updates: {len(errors)}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.log_folder + '\\' + filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info("Plot saved as %s", filename)


    def plot_opinion_evolution(self, num_users = 4, filename: str = "opinion_evolution.png"):
        """Plot the evolution of select opinions."""

        opinions = self.data.loc[:,"opinion_state"]
        steps = self.data.loc[:, "step"]
        assert len(opinions) > 0, "No opinions to plot"
        
        # Parse the opinion arrays (handle both string and array formats)
        opinion_arrays = []
        for opinion_data in opinions:
                opinion_arrays.append(opinion_data)
        
        opinion_matrix = np.array(opinion_arrays)

        n_timesteps, n_users_total = opinion_matrix.shape
        
        # Select random users to plot
        selected_users = np.random.choice(n_users_total, 
                                        size=min(num_users, n_users_total), 
                                        replace=False)
        selected_users = np.sort(selected_users)  # Sort for consistent ordering
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot a dashed line for the steady state we are trying to keep
        steady_state = self.key_stats.get("steady_state", np.zeros(n_users_total))
        ax.axhline(y=steady_state.mean(), color='gray', linestyle='--', label='Steady State', alpha=0.7)
        
        # Plot each selected user's opinion evolution
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_users)))
        for i, user_idx in enumerate(selected_users):
            user_opinions = opinion_matrix[:, user_idx]
            ax.plot(steps, user_opinions, 
                   color=colors[i], 
                   linewidth=2, 
                   marker='o', 
                   markersize=3,
                   alpha=0.8,
                   label=f'User {user_idx}'
                   )
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Opinion Value')
        ax.set_title(f'Opinion Evolution for {len(selected_users)} Selected Users')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(-1, 1)
        
        # Add some statistics
        final_opinions = opinion_matrix[-1, selected_users]
        initial_opinions = opinion_matrix[0, selected_users]
        opinion_changes = final_opinions - initial_opinions
        
        # Add text box with statistics
        stats_text = f"""
        Selected Users: {selected_users.tolist()}
        Initial Opinions: {initial_opinions}
        Final Opinions: {final_opinions}
        Opinion Changes: {opinion_changes}
        Max Change: {np.max(np.abs(opinion_changes)):.4f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.log_folder + '\\' + filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info("Plot saved as %s", filename)
