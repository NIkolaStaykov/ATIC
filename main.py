from omegaconf import OmegaConf
import hydra
from data_generator import DataGenerator

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg):
    generator = DataGenerator(cfg['data_generation'])
    file = open("output.csv", "w", encoding="utf-8")
    file.write("step,opinion_state,control_input,attacker_input\n")
    for data in generator.generate():
        file.write(f"{data['step']},{data['opinion_state']},{data['control_input']},{data['attacker_input']}\n")
    generator.plot_estimation_error()

if __name__ == "__main__":
    # The @hydra.main decorator handles calling my_app()
    my_app()
