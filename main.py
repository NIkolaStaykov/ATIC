from omegaconf import OmegaConf
import hydra
from data_generator import DataGenerator
from data_analysis import Plotter
import pandas as pd
import numpy as np
import os

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg):
    """Where it all comes together."""

    generator = DataGenerator(cfg['data_generation'])
    columns = ["step", "opinion_state", "control_input", "attacker_input", "sensitivity_estimate", "kalman_covariance_trace", "estimation_error"]
    dataset = pd.DataFrame(columns=columns)
    # In this dict we can intermediate results from the generator to avoid recalculating them
    key_data_stats = {}

    # Append to existing CSV file or create a new one
    filename = "final_steady_state_opinions.csv"
    if not os.path.exists(filename):
        file = open(filename, "w", encoding="utf-8")
        file.write("seed,opinion_state_avg,opinion_state_var,steady_state,final_kalman_err\n")
    else:
        file = open(filename, "a", encoding="utf-8")

    for data in generator.generate():
        # extract key statistics from the data
        key_data_stats.update(data.get("key_stats", {}))
        dataset = pd.concat([dataset, pd.DataFrame([data], columns=dataset.columns)], axis=0,  ignore_index=True)
    
    # We assume the opinions converged on the last step
    last_row = dataset.iloc[-1]
    opinion_avg = last_row['opinion_state'].mean()
    opinion_var = last_row['opinion_state'].var()
    seed = cfg['data_generation']['seed']
    final_kalman_err = last_row['estimation_error']
    steady_state = key_data_stats.get("steady_state", np.zeros(cfg['data_generation']['network']['num_users'])).mean()
    file.write(f"{seed},{opinion_avg},{opinion_var},{steady_state},{final_kalman_err}\n")
    file.close()

    # Plotting
    plotter = Plotter(dataset, key_data_stats)
    plotter.plot_estimation_error()
    plotter.plot_opinion_evolution()

if __name__ == "__main__":
    # The @hydra.main decorator handles calling my_app()
    my_app()
