import os
import sys

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

from tools.config_file import NewUserPredictParams
params = NewUserPredictParams()

import pandas as pd


if __name__ == "__main__":
    print("Conformity result:")
    df_result1 = pd.read_csv("./result/2023_09_09_17_21_LGBM_result.csv")
    df_result2 = pd.read_csv("./result/2023_09_09_03_21_LGBM_result.csv")
    cnt = 0
    for i in range(len(df_result1)):
        if df_result1['target'][i] or df_result2['target'][i]:
            df_result1['target'][i] = 1
            print("Find a target:")
            cnt += 1
    print(f"Finish conformity: {cnt}")
    df_result1.to_csv("./result/conformity.csv", index=False)