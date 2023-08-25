import os

import pandas
import yaml


class NewUserPredictParams:
    def __init__(self):
        self.__PROJECT_DIR__ = "E:\\project\\NewUserPredict"

        with open(os.path.join(self.__PROJECT_DIR__, "tools", "config.yaml"), "r") as f:
            config_file = yaml.safe_load(f)

        # Train dataset absolute path
        self.train_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_csv"])
        self.train_all_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_all_pt"])
        # Test dataset absolute path

        # Train dataset which has been divided into two parts
        self.train_unknown_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_unknown_csv"])
        self.train_unknown_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_unknown_pt"])

        # Test dataset which has been divided into two parts
        # Maybe this process is useless
        self.train_known_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_known_csv"])
        self.train_known_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_known_pt"])

        self.test_csv = os.path.join(self.__PROJECT_DIR__, config_file["test_csv"])
        self.test_all_pt = os.path.join(self.__PROJECT_DIR__, config_file["test_all_pt"])

        self.result_all_csv = os.path.join(self.__PROJECT_DIR__, config_file["result_all_csv"])

        # Model
        self.model_save_path = os.path.join(self.__PROJECT_DIR__, config_file["model_save_path"])
        self.plt_save_path = os.path.join(self.__PROJECT_DIR__, config_file["plt_save_path"])

        # Load
        self.division_rate = config_file["division_rate"]

        self.input_size = config_file["input_size"]
        self.hidden_size1 = config_file["hidden_size1"]
        self.hidden_size2 = config_file["hidden_size2"]
        self.lr = config_file["lr"]
        self.num_epochs = config_file["num_epochs"]
        self.batch_size = config_file["batch_size"]

        self.we_com_webhook_url = config_file["we_com_webhook_url"]

        # Train dataset and test dataset which has complete normalization
        self.train_norm_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_norm_pt"])
        self.test_norm_pt = os.path.join(self.__PROJECT_DIR__, config_file["test_norm_pt"])


if __name__ == "__main__":
    params = NewUserPredictParams()
    print(params.train_csv)
    test_csv = pandas.read_csv(params.train_csv)
    print(test_csv)
