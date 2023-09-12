import os
import sys

import colorama
current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

from flaml import AutoML
from ray import tune
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import pandas as pd
from datetime import datetime
import numpy as np
from tools.load_data_automl import load_automl
from tools.config_file import NewUserPredictParams
from utils_webhook.WeCom import WeCom

config_params = NewUserPredictParams()


def train_automl(features: np.ndarray, target: np.ndarray) -> AutoML:
    # 创建 AutoML 对象
    automl = AutoML()

    # 设置 AutoML 配置，包括任务类型和评估指标
    automl_settings = {
        "task": "classification",  # 分类任务
        "metric": "f1",  # 评估指标为 F1 分数
        "time_budget": 60 * 60 * 2,
        "estimator_list": ["xgboost"],
        "n_jobs": -1,
    }

    # 使用 AutoML 进行训练和优化
    automl.fit(X_train=features, y_train=target, **automl_settings)
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparameter config:', automl.best_config)
    print('Best accuracy on validation data: {0:.4g}'.format(1 - automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
    return automl


def save_best_to_pkl(model: AutoML):
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # 构建保存文件的路径，将时间作为文件名的一部分
    model_save_path = os.path.join(config_params.automl_model_save_dir, f"{current_time}_LGBM.pkl")
    # 保存模型到文件
    joblib.dump(model, model_save_path)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Save OK!")
    print(colorama.Fore.RESET)


if __name__ == "__main__":
    """
    Notice: In every single training, the training dataset is different
    """
    x_train, x_test, y_train, y_test = load_automl(config_params.train_processed_csv, config_params.test_processed_csv,
                                                   True)
    automl = train_automl(x_train, y_train)

    y_pred = automl.predict(x_test)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    content = f"""
    Precision: {precision}\n
    Recall: {recall}\n
    F1_score: {f1}\n
    """
    print(content)

    save_best_to_pkl(automl)
    weCom = WeCom(config_params.we_com_webhook_url)
    weCom.generate_md(content)
    weCom.send()