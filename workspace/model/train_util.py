import time
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from workspace.model.model import model_path
from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list, combine
from workspace.util.combine_dataset import encode_df

from sklearn.metrics import r2_score


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def check_running_time(message: str, st_time: time.time) -> time.time:
    print(f'{message}, Running time: {time.time() - st_time}')
    return time.time()


def save_model(model, optim, path, file_name):
    state = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }
    torch.save(state, f'{path}/{file_name}')


def load_model(model, optim, path, file_name):
    state = torch.load(f'{path}/{file_name}')

    model.load_state_dict(state["model_state"])

    if optim is not None:
        optim.load_state_dict(state["optim_state"])
        return model, optim

    return model


def predict_reg(model, dl, device):
    with torch.no_grad():
        model.eval()

        total_y_test = np.array(())
        total_yhat = np.array(())

        for X_test, y_test in dl:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            yhat = model(X_test)

            total_y_test = np.append(total_y_test, y_test.detach().cpu())
            total_yhat = np.append(total_yhat, yhat.detach().cpu())

        r2 = r2_score(y_test.detach().cpu(), yhat.detach().cpu())

    return r2


def reg_training(model, optim_func, loss_func, train_dl, test_dl, val_dl, lr, model_name, target_col, device='cpu',
                 epochs=1000, is_load=True):
    st_time = time.time()

    print_epoch = epochs / 50

    create_dir(f'{model_path}/{model_name}')
    train_log = os.listdir(f'{model_path}/{model_name}')

    optim = optim_func(model.parameters(), lr=lr, weight_decay=0.1)

    if len(train_log) > 0 and is_load:
        last_train_epoch = train_log[-1].split('_')[-1].split('.')[0]
        print(f'start at epoch({last_train_epoch})')
        model, optim = load_model(model, optim, f'{model_path}/{model_name}', f"{target_col}_{last_train_epoch}.pt")
        last_train_epoch = int(last_train_epoch)
        print(f'load {model_name}/{target_col}_{last_train_epoch}.pt')
    else:
        last_train_epoch = 0

    model = model.to(device)

    for epoch in range(last_train_epoch, epochs):
        # X_batch: batch, sequence, feature
        for X_batch, y_batch in train_dl:
            model.train()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            out = model(X_batch)
            y_batch = y_batch.squeeze()
            out = out.squeeze()
            loss = loss_func(out, y_batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch % print_epoch == 0:
            print(f'{epoch}/{epochs}: loss: {loss}')
            st_time = check_running_time('', st_time)
            model.eval()
            r2 = predict_reg(model, test_dl, device)

            print(f'eval r2: {r2}')

        if epoch % (print_epoch * 2) == 0:
            print('save models')
            save_model(model, optim, f'{model_path}/{model_name}', f"{target_col}_{epoch}.pt")
    save_model(model, optim, f'{model_path}/{model_name}', f"{target_col}_{epochs}.pt")

    if val_dl:
        st_time = check_running_time('start validation', st_time)
        with torch.no_grad():
            model.eval()
            r2 = predict_reg(model, val_dl, device)

        check_running_time(f'validation r2: {r2}', st_time)
