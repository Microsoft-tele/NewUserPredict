import os
import sys

import colorama
import pandas
import yaml


class NewUserPredictParams:
    def __init__(self):
        current_file = os.path.abspath(__file__)
        parent_dir = os.path.dirname(current_file)
        great_parent_dir = os.path.dirname(parent_dir)

        self.__PROJECT_DIR__ = great_parent_dir
        sys.path.append(self.__PROJECT_DIR__)

        with open(os.path.join(self.__PROJECT_DIR__, "tools", "config.yaml"), "r") as f:
            config_file = yaml.safe_load(f)

        # Train dataset absolute path
        self.train_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_csv"])
        self.train_processed_csv = os.path.join(self.__PROJECT_DIR__, config_file["train_processed_csv"])
        self.train_pt = os.path.join(self.__PROJECT_DIR__, config_file["train_pt"])

        # Train dataset via being processing by classify.py
        self.train_classified_csv = [
            os.path.join(self.__PROJECT_DIR__, config_file["train_2_3_csv"]),
            os.path.join(self.__PROJECT_DIR__, config_file["train_3_csv"]),
            os.path.join(self.__PROJECT_DIR__, config_file["train_1_2_3_4_5_csv"]),
            os.path.join(self.__PROJECT_DIR__, config_file["train_unknown_csv"])
        ]
        self.train_classified_pt = [
            os.path.join(self.__PROJECT_DIR__, config_file["train_2_3_pt"]),
            os.path.join(self.__PROJECT_DIR__, config_file["train_3_pt"]),
            os.path.join(self.__PROJECT_DIR__, config_file["train_1_2_3_4_5_pt"]),
            os.path.join(self.__PROJECT_DIR__, config_file["train_unknown_pt"])
        ]
        
        self.test_classified_csv = [
            os.path.join(self.__PROJECT_DIR__, config_file["test_2_3_csv"]),
            os.path.join(self.__PROJECT_DIR__, config_file["test_3_csv"]),
            os.path.join(self.__PROJECT_DIR__, config_file["test_1_2_3_4_5_csv"]),
            os.path.join(self.__PROJECT_DIR__, config_file["test_unknown_csv"])
        ]
        self.test_classified_pt = [
            os.path.join(self.__PROJECT_DIR__, config_file["test_2_3_pt"]),
            os.path.join(self.__PROJECT_DIR__, config_file["test_3_pt"]),
            os.path.join(self.__PROJECT_DIR__, config_file["test_1_2_3_4_5_pt"]),
            os.path.join(self.__PROJECT_DIR__, config_file["test_unknown_pt"])
        ]

        self.test_csv = os.path.join(self.__PROJECT_DIR__, config_file["test_csv"])
        self.test_processed_csv = os.path.join(self.__PROJECT_DIR__, config_file["test_processed_csv"])
        self.test_pt = os.path.join(self.__PROJECT_DIR__, config_file["test_pt"])

        # Model
        self.model_save_path = os.path.join(self.__PROJECT_DIR__, config_file["model_save_path"])
        self.plt_save_path = os.path.join(self.__PROJECT_DIR__, config_file["plt_save_path"])
        self.result_save_path = os.path.join(self.__PROJECT_DIR__, config_file["result_save_path"])

        # Load
<<<<<<< HEAD
        self.division_rate = config_file["division_rate"]
        self.input_size = config_file["input_size"]
        self.hidden_size1 = config_file["hidden_size1"]
        self.hidden_size2 = config_file["hidden_size2"]
        self.lr = config_file["lr"]
        self.num_epochs = config_file["num_epochs"]
        self.batch_size = config_file["batch_size"]
        self.threshold = config_file["threshold"]

=======
>>>>>>> 27f50b410b1ef7a331bfacc63d47d524e90668e4
        self.we_com_webhook_url = config_file["we_com_webhook_url"]


if __name__ == "__main__":
    params = NewUserPredictParams()
    print(params.train_csv)
    test_csv = pandas.read_csv(params.train_csv)
    print(test_csv)
