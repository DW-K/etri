import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import life_label_Dataset, VanillaRnn, GRU, GRU_bi
from train_util import reg_training
from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list
from workspace.util.getData import get_data, preprocessing

if __name__ == "__main__":
    batch_size = 256
    sequence_size = 5
    interval = 1

    user_list = get_user_list()[:-2]
    ul = user_label_encoder()

    target_col = 'emotionPositive'

    drop_col = ['mAcc_x', 'mAcc_y', 'mAcc_z', 'mGps_lat', 'mGps_lon', 'mGps_accuracy', 'mGyr_x', 'mGyr_y', 'mGyr_z',
                'mGyr_roll', 'mGyr_pitch', 'mGyr_yaw', 'mMag_x', 'mMag_y', 'mMag_z']

    X_array, y_array = get_data(user_list, target_col, sequence_size, interval, drop_col)

    X_train, y_train, X_test, y_test, X_val, y_val = preprocessing(X_array, y_array)

    train_ds = life_label_Dataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = life_label_Dataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    val_ds = life_label_Dataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    num_layers = 5
    lr = 0.001
    epochs = 340

    input_size = X_array.shape[-1]
    hidden_size = 128
    # num_classes = ul.get_label_size(target_col)
    num_classes = 1

    model = GRU_bi(input_size=input_size, hidden_size=hidden_size, sequence_size=sequence_size,
                       num_layers=num_layers, num_classes=num_classes, device=device).to(device)

    optim_func = torch.optim.Adam
    # loss_func = F.cross_entropy
    loss_func = F.mse_loss

    print(f'train size: {train_ds.__len__()}, test size: {test_ds.__len__()}, val size: {val_ds.__len__()}')
    print(f'hidden_size: {hidden_size}, sequence_size: {sequence_size}, num_layers: {num_layers}')

    # def training(model, optim, loss_func, train_dl, test_dl, model_name, target_col, version, device='cpu',
    # epochs=1000, print_epoch=50):
    reg_training(model=model, optim_func=optim_func, loss_func=loss_func, train_dl=train_dl, test_dl=test_dl,
                 val_dl=val_dl, lr=lr,
                 model_name='bi_gru_e4_hidden128_reg', target_col=target_col, device=device, epochs=epochs)
