import pandas as pd


def statistic_certain_column(df_dataset: pd.DataFrame, column: str):
    """
    统计给定的 column 的平均数，中位数，众数，标准差和方差
    Args:
        df_dataset: 数据框
        column: 列名

    Returns:
        无，打印统计结果
    """
    col_data = df_dataset[column]
    mean_value = col_data.mean()
    median_value = col_data.median()
    mode_value = col_data.mode().iloc[0]  # mode() returns a Series, so we use iloc[0] to get the mode value
    std_value = col_data.std()
    var_value = col_data.var()

    print(f"Column: {column}")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Mode: {mode_value}")
    print(f"Standard Deviation: {std_value}")
    print(f"Variance: {var_value}")
