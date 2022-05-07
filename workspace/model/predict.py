import os

import torch
from torch.utils.data import DataLoader

from workspace.model.model import life_label_Dataset, model_path
from workspace.model.train_util import load_model, predict_reg
from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list
from workspace.util.getData import get_data, preprocessing


def get_dl():
    user_list = get_user_list()[:-2]

    target_col = 'emotionPositive'
    sequence_size = 5
    interval = 1
    batch_size = 256

    drop_col = ['mAcc_x', 'mAcc_y', 'mAcc_z', 'mGps_lat', 'mGps_lon', 'mGps_accuracy', 'mGyr_x', 'mGyr_y', 'mGyr_z', 'mGyr_roll',
     'mGyr_pitch', 'mGyr_yaw', 'mMag_x', 'mMag_y', 'mMag_z']

    X_array, y_array = get_data(user_list, target_col, sequence_size, interval, drop_col)

    X_train, y_train, X_test, y_test, X_val, y_val = preprocessing(X_array, y_array)

    train_ds = life_label_Dataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = life_label_Dataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    val_ds = life_label_Dataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return val_dl


def predict(model_name, model_func):
    sequence_size = 5

    target_col = 'emotionPositive'

    val_dl = get_dl()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = 5

    input_size = 22
    hidden_size = input_size * 2
    num_classes = 1

    model = model_func(input_size=input_size, hidden_size=hidden_size, sequence_size=sequence_size,
                       num_layers=num_layers, num_classes=num_classes, device=device).to(device)

    train_log = os.listdir(f'{model_path}/{model_name}')
    last_train_epoch = train_log[-1].split('_')[-1].split('.')[0]
    model = load_model(model, None, f'{model_path}/{model_name}', f"{target_col}_{last_train_epoch}.pt")

    r2 = predict_reg(model, val_dl, device)

    print(r2)
