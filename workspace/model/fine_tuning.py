import time
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from workspace.model.model import model_path, life_label_Dataset, VanillaRnn
from workspace.model.train_util import load_model, predict_reg, reg_training
from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list, combine
from workspace.util.combine_dataset import encode_df

from sklearn.metrics import r2_score

from workspace.util.getData import preprocessing, get_data


def fine_tunging(model_name, epochs, user_num, model_func):
    batch_size = 256
    sequence_size = 5
    interval = 1

    target_col = 'emotionPositive'

    drop_col = ['mAcc_x', 'mAcc_y', 'mAcc_z', 'mGps_lat', 'mGps_lon', 'mGps_accuracy', 'mGyr_x', 'mGyr_y', 'mGyr_z',
                'mGyr_roll',
                'mGyr_pitch', 'mGyr_yaw', 'mMag_x', 'mMag_y', 'mMag_z']

    X_array, y_array = get_data(user_num, target_col, sequence_size, interval, drop_col)

    idx_range = np.arange(1, X_array.shape[0], 60)

    print(X_array.shape)

    X_array = X_array[idx_range, :]
    y_array = y_array[idx_range]

    X_train, y_train, X_test, y_test, X_val, y_val = preprocessing(X_array, y_array)

    print(X_train.shape, X_test.shape, X_val.shape)

    train_ds = life_label_Dataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = life_label_Dataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    val_ds = life_label_Dataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = 5
    lr = 0.001

    input_size = X_array.shape[-1]
    hidden_size = input_size * 2
    # num_classes = ul.get_label_size(target_col)
    num_classes = 1

    optim_func = torch.optim.Adam
    loss_func = F.mse_loss

    model = model_func(input_size=input_size, hidden_size=hidden_size, sequence_size=sequence_size,
                       num_layers=num_layers, num_classes=num_classes, device=device).to(device)

    optim = optim_func(model.parameters(), lr=lr, weight_decay=0.1)

    train_log = os.listdir(f'{model_path}/{model_name}')
    last_train_epoch = train_log[-1].split('_')[-1].split('.')[0]
    print(f'start at epoch({last_train_epoch})')
    model, optim = load_model(model, optim, f'{model_path}/{model_name}', f"{target_col}_{last_train_epoch}.pt")
    last_train_epoch = int(last_train_epoch)
    print(f'load {model_name}/{target_col}_{last_train_epoch}.pt')

    model = load_model(model, None, f'{model_path}/{model_name}', f"{target_col}_{last_train_epoch}.pt")

    print('-------------------------------')
    print('orgin model(validation)')
    r2 = predict_reg(model, val_dl, device)
    print(r2)
    print('-------------------------------')

    print(f'train size: {train_ds.__len__()}, test size: {test_ds.__len__()}')
    print(f'hidden_size: {hidden_size}, sequence_size: {sequence_size}, num_layers: {num_layers}')

    reg_training(model=model, optim_func=optim_func, loss_func=loss_func, train_dl=train_dl, test_dl=test_dl,
                 val_dl=val_dl, lr=lr,
                 model_name=f'{model_name}_fine_{user_num}', target_col=target_col, device=device, epochs=epochs,
                 is_load=True)

    model_fine = model_func(input_size=input_size, hidden_size=hidden_size, sequence_size=sequence_size,
                            num_layers=num_layers, num_classes=num_classes, device=device).to(device)

    model_fine = load_model(model_fine, None, f'{model_path}/{model_name}_fine_{user_num}', f"{target_col}_{epochs}.pt")

    print('-------------------------------')
    print('fine tuning model(validation)')
    r2 = predict_reg(model_fine, val_dl, device)
    print(r2)
    print('-------------------------------')