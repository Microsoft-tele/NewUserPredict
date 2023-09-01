from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def normalize_by_columns(pd_dataset: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalize selected columns in a Pandas DataFrame using MinMaxScaler.

    Args:
        pd_dataset (pd.DataFrame): The dataset to be normalized.
        columns (list): Selected columns to be normalized.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    normalized_dataset = pd_dataset.copy()  # Create a copy to avoid modifying the original DataFrame

    scaler = MinMaxScaler()  # Create a MinMaxScaler instance

    for column in columns:
        # Fit and transform the selected column using MinMaxScaler
        normalized_dataset[column] = scaler.fit_transform(normalized_dataset[[column]])

    return normalized_dataset


def standardize_by_columns(pd_dataset: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Standardize selected columns in a Pandas DataFrame using MinMaxScaler.

    Args:
        pd_dataset (pd.DataFrame): The dataset to be normalized.
        columns (list): Selected columns to be normalized.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df_standardized = pd_dataset.copy()  # Create a copy to avoid modifying the original DataFrame

    scaler = StandardScaler()  # Create a MinMaxScaler instance

    for column in columns:
        # Fit and transform the selected column using MinMaxScaler
        df_standardized[column] = scaler.fit_transform(df_standardized[[column]])

    return df_standardized