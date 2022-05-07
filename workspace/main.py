from workspace.model.fine_tuning import fine_tunging
from workspace.model.model import GRU_bi
from workspace.model.predict import predict
from workspace.model.train import train
from workspace.util.User import get_user_list

if __name__ == "__main__":
    batch_size = 256
    sequence_size = 5
    interval = 1

    user_list = get_user_list()[:-2]

    target_col = 'embaotionPositive'

    drop_col = ['mAcc_x', 'mAcc_y', 'mAcc_z', 'mGps_lat', 'mGps_lon', 'mGps_accuracy', 'mGyr_x', 'mGyr_y', 'mGyr_z',
                'mGyr_roll', 'mGyr_pitch', 'mGyr_yaw', 'mMag_x', 'mMag_y', 'mMag_z']

    num_layers = 5
    lr = 0.001
    epochs = 340
    hidden_size = 128

    model_func = GRU_bi

    model_name = "GRU_bi"

    train(batch_size, sequence_size, interval, user_list, target_col, drop_col, num_layers, hidden_size, lr, epochs, model_func, model_name)

    predict(model_name, model_func)

    user_num = '30'

    fine_tunging(model_name, epochs, user_num, model_func)
