import os
import sys

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from tools.config_file import NewUserPredictParams
from tools.load_data_automl import load_automl

config_params = NewUserPredictParams()


def select_automl_model(root_dir: str) -> str:
    # 获取目录下的所有文件
    files = os.listdir(root_dir)

    # 筛选出所有以 ".pkl" 结尾的文件
    pkl_files = [file for file in files if file.endswith(".pkl")]

    if not pkl_files:
        print("在目录中没有找到任何 .pkl 文件。")
        return None

    # 打印所有 .pkl 文件并要求用户选择
    print("可用的 .pkl 模型文件：")
    for i, file in enumerate(pkl_files):
        print(f"{i + 1}: {file}")

    # 要求用户输入选择的序号
    while True:
        try:
            choice = int(input("请输入要选择的模型文件的序号："))
            if 1 <= choice <= len(pkl_files):
                selected_file = pkl_files[choice - 1]
                return os.path.join(root_dir, selected_file)
            else:
                print("请输入有效的序号。")
        except ValueError:
            print("请输入有效的序号。")


if __name__ == "__main__":
    best_model: AutoML = joblib.load(select_automl_model(config_params.automl_model_save_dir))
    np_features = load_automl(config_params.train_processed_csv, config_params.test_processed_csv,
                              is_train=False)

    y_pred = best_model.predict(np_features)
    df_result = pd.DataFrame(y_pred)
    df_result = df_result.rename(columns={0: 'target'})
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    df_result.to_csv(os.path.join(config_params.result_save_path, f"{current_time}_LGBM_result.csv"))
    print("Save OK")
