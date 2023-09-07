import pandas as pd
from sklearn.model_selection import train_test_split


def load_automl(train_path: str, test_path: str, is_train: bool):
    features = [
        'eid', 'eid_target', 'common_ts', 'date', 'hour', 'weekday', 'sin_norm', 'cos_norm', 'sin', 'cos', 'x1', 'x2',
        'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x1_frequency', 'x2_frequency', 'x6_frequency', 'x8_frequency']
    if is_train:
        # return train
        """
        [uuid,eid,udmap,common_ts,x1,x2,x3,x4,x5,x6,x7,x8,key1,key2,key3,key4,key5,key6,key7,key8,key9,target,
        common_ts_dt,date,hour,weekday,sin_norm,cos_norm,sin,cos,eid_target]
        """
        df_train = pd.read_csv(train_path)
        df_features = df_train[features]
        pd.set_option('display.max_columns', None)
        print(df_features)
        np_features = df_features.values
        print("Shape of features:", df_features.shape)
        df_target = df_train.pop('target')
        np_target = df_target.values
        print("Shape of target:", df_target.shape)

        x_train, x_test, y_train, y_test = train_test_split(np_features, np_target, test_size=0.1, random_state=42)

        return x_train, x_test, y_train, y_test

    else:
        # return test
        df_test = pd.read_csv(test_path)
        df_features = df_test[features]
        pd.set_option('display.max_columns', None)
        print(df_features)
        np_features = df_features.values
        print("Shape of features:", df_features.shape)
        return np_features
