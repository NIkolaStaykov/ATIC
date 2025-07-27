from omegaconf import OmegaConf
import hydra
from data_generator import DataGenerator
from data_analysis import Plotter
import pandas as pd

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg):
    """Where it all comes together."""

    generator = DataGenerator(cfg['data_generation'])
    columns = ["step", "opinion_state", "control_input", "attacker_input", "sensitivity_estimate", "kalman_covariance_trace", "estimation_error"]
    dataset = pd.DataFrame(columns=columns)

    file = open("output.csv", "w", encoding="utf-8")
    file.write("step,opinion_state,control_input,attacker_input\n")
    for data in generator.generate():
        dataset = pd.concat([pd.DataFrame([data], columns=dataset.columns), dataset], axis=0,  ignore_index=True)
        file.write(f"{data['step']},{data['opinion_state']},{data['control_input']},{data['attacker_input']}\n")
    
    plotter = Plotter(dataset)
    plotter.plot_estimation_error()
    plotter.plot_opinion_evolution()

if __name__ == "__main__":
    # The @hydra.main decorator handles calling my_app()
    my_app()
