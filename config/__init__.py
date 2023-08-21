import os

import pandas
import yaml


class Config:
    def __init__(self):
        self.__PROJECT_DIR__ = os.path.abspath(os.path.join(os.getcwd(), ".."))

        with open(os.path.join(self.__PROJECT_DIR__, "config", "config.yaml"), "r") as f:
            config_file = yaml.safe_load(f)

        # Train dataset absolute path
        self.train_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_csv"])
        # Test dataset absolute path

        # Train dataset which has been divided into two parts
        self.train_unknown_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_unknown_csv"])
        self.train_unknown_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_unknown_pt"])

        # Test dataset which has been divided into two parts
        # Maybe this process is useless
        self.train_known_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_known_csv"])
        self.train_known_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_known_pt"])

        # Load
        self.division_rate = config_file["division_rate"]

        self.input_size = config_file["input_size"]
        self.hide_size = config_file["hide_size"]
        self.lr = config_file["lr"]
        self.num_epochs = config_file["num_epochs"]
        self.batch_size = config_file["batch_size"]


if __name__ == "__main__":
    config = Config()
    print(config.train_csv)
    test_csv = pandas.read_csv(config.train_csv)
    print(test_csv)
