import os
import traceback
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list, combine
from workspace.util.combine_dataset import encode_df, check_running_time

dataset_path = "../../dataset"
model_path = "./models"


def get_data(user_list, target_col, sequence_size, interval=1, drop_col=None):
    if drop_col is None:
        drop_col = ['mGps_lat', 'mGps_lon', 'mGps_accuracy']

    st_time = time.time()

    if os.path.exists(dataset_path + f"/seq/interval{interval}_seq{sequence_size}_user{len(user_list)}_X.npy"):
        X = np.load(dataset_path + f"/seq/interval{interval}_seq{sequence_size}_user{len(user_list)}_X.npy")

        if os.path.exists(dataset_path + f"/seq/{target_col}_interval{interval}_seq{sequence_size}_user{len(user_list)}_y.npy"):
            y = np.load(dataset_path + f"/seq/{target_col}_interval{interval}_seq{sequence_size}_user{len(user_list)}_y.npy")

            check_running_time("load data", st_time)

            return X, y

    if type(user_list) is not list:
        user_list = [user_list]

    total_user_list = get_user_list()

    for user in user_list:
        if user not in total_user_list:
            print('user num error')
            raise

    X = np.array([])
    y = np.array([])

    ul = user_label_encoder()
    encoding_label_list = ul.get_label()

    for user in user_list:
        # get data
        print(f'get user{user}')
        co = combine(user, interval)
        df = co.get_combine(target_col)
        # print(f'----{user}----')
        # print(df.shape)

        # drop GPS
        df = df.drop(drop_col, axis=1)

        # NAN
        df_dropna = df.dropna(how='any', axis=0).copy()
        print(f'-----NAN droped {df.shape[0] - df_dropna.shape[0]}/{df.shape[0]}')

        # encoding
        if target_col in encoding_label_list:
            try:
                df_dropna.loc[:, target_col] = encode_df(df_dropna.loc[:, target_col], target_col)
            except Exception as e:
                print(f'user{user}: {traceback.format_exc()}')
                raise

        df_drop_ts = df_dropna.drop('timestamp', axis=1)

        temp_X = df_drop_ts.drop(target_col, axis=1).values
        temp_y = df_drop_ts.loc[:, target_col].values

        temp_X_list = []
        temp_y_list = []
        for idx in range(temp_X.shape[0] - sequence_size):
            temp_X_list.append(temp_X[idx:idx + sequence_size, :])
            temp_y_list.append(temp_y[idx + sequence_size])

        temp_seq_X = np.stack(temp_X_list)
        temp_seq_y = np.stack(temp_y_list)

        if X.shape[-1] == 0:
            X = X.reshape(0, temp_seq_X.shape[1], temp_seq_X.shape[2])

        if temp_seq_X.shape[-1] != X.shape[-1]:
            print(f'user{user} has diff col size, X: {X.shape}, but {temp_seq_X.shape}')

        else:
            X = np.concatenate((X, temp_seq_X), 0)
            y = np.concatenate((y, temp_seq_y), 0)

        print(f'X: {X.shape}, temp: {temp_seq_X.shape}')
        print(f'y: {y.shape}, temp: {temp_seq_y.shape}')

    if not os.path.exists(dataset_path + f"/seq/interval{interval}_seq{sequence_size}_user{len(user_list)}_X.npy"):
        np.save(dataset_path + f"/seq/interval{interval}_seq{sequence_size}_user{len(user_list)}_X.npy", X)

    np.save(dataset_path + f"/seq/{target_col}_interval{interval}_seq{sequence_size}_user{len(user_list)}_y.npy", y)

    check_running_time("get & make dataset", st_time)

    return X, y


def preprocessing(X, y, is_val=True):
    np.random.seed(42)
    # 0.6 : 0.2 : 0.2
    train_num = int(X.shape[0] * 0.6)
    test_num = int(X.shape[0] * 0.2)

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:train_num], indices[train_num:]

    if is_val:
        test_idx, val_idx = indices[train_num:train_num + test_num], indices[train_num + test_num:]

    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]

    if is_val:
        X_val = X[val_idx, :]
        y_val = y[val_idx]

    ss_X = StandardScaler()
    ss_X.fit(X_train.reshape(-1, X_train.shape[-1]))

    X_train_scaled = ss_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = ss_X.transform(X_test.reshape(-1, X_train.shape[-1])).reshape(X_test.shape)

    if is_val:
        X_val_scaled = ss_X.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)
        return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val

    return X_train_scaled, y_train, X_test_scaled, y_test