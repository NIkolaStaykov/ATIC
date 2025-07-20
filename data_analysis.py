import matplotlib.pyplot as plt
import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class Plotter:
    def __init__(self, data: pd.DataFrame):
        self.log = logging.getLogger(f"\033[96m[{self.__class__.__name__}]\033[0m")
        self.data = data

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
        
        Final Error: {errors.iloc[-1]:.6f}
        Mean Error: {np.mean(errors):.6f}
        Min Error: {np.min(errors):.6f}
        Max Error: {np.max(errors):.6f}
        Std Error: {np.std(errors):.6f}
        
        Total Updates: {len(errors)}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info("Plot saved as %s", filename)
