from src.visualization.viz_plots import viz_plots
from src.features.auto_preprocess import auto_preprocess
from src.models.train_and_save import train_and_save
import configs
import random
import os
import numpy as np


def seed_everything(seed=configs.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    seed_everything()
    viz_plots(configs.TRAIN_PATH, configs.MODELS_OUTPUT)
    auto_preprocess(
        configs.TRAIN_PATH, configs.TEST_PATH, configs.DROP_COLS, configs.TARGET
    )
    train_and_save(configs.NEW_TRAIN_PATH, configs.MODELS_OUTPUT)
