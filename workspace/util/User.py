import os
import pandas as pd
from datetime import datetime
import time
from bisect import bisect

from workspace.util.combine_dataset import get_sensor_label

dataset_path = r'../../dataset'


# datetime을 timestamp로
def mk_ts(d):
    return int(d.timestamp())


# timestamp를 datetime으로
def mk_dt(ts):
    return datetime.fromtimestamp(ts)


def get_total_date_list():
    path = r'./user/total_date.txt'

    df = pd.read_csv(path, header=None)

    total_date_list = list(map(mk_dt, df.values[0].astype(int).tolist()))

    return total_date_list


def get_category_list():
    category_list = ['e4Acc', 'e4Bvp', 'e4Eda', 'e4Hr', 'e4Temp',
                     'mAcc', 'mGps', 'mGyr', 'mMag']

    return category_list


def get_label_col_list():
    label_col_list = ['action', 'actionOption', 'actionSubOption', 'condition', 'conditionSub1Option',
                      'conditionSub2Option', 'place', 'emotionPositive', 'emotionTension', 'activity']

    return label_col_list


def get_user_list():
    user_list = [str(i) if i >= 10 else '0' + str(i) for i in range(1, 11 + 1)]
    user_list += [str(i) if i >= 10 else '0' + str(i) for i in range(21, 30 + 1)]

    return user_list


class lifelog:
    # date input은 항상 datetime object
    def __init__(self, user_num):
        self.path = dataset_path + f'/lifelog/2020/user{user_num}'
        self.user_num = user_num

        self.date_list = list(
            map(datetime.fromtimestamp,
                map(int, os.listdir(self.path))  # path를 받아오는 거니까 string을 int로 mapping
                )
        )
        self.date_list.sort()

        self.category_list = get_category_list()

    # s_day: start day | e_day: end day
    # 특정 datetime의 첫번째 index timestamp부터 마지막 index timestamp까지
    def get_sensor(self, category: str, s_day: datetime, e_day: datetime = None):
        if e_day is None:
            e_day = s_day

        date_start_index = bisect(self.date_list, s_day)
        date_end_index = bisect(self.date_list, e_day)

        df = pd.DataFrame()

        if category in self.category_list or category == 'label':
            for i in range(date_start_index - 1, date_end_index):
                df = df.append(
                    self.get_sensor_data(category, self.date_list[i])  # 이건 아래에 있는 function
                )

            df = df.reset_index(drop=True)
            return df
        else:
            print('wrong category')

    def get_sensor_data(self, category: str, day: datetime):
        day_ts = mk_ts(day)
        path = self.path + f'/{day_ts}' + f'/{day_ts}_{category}' + '.csv'

        df = pd.read_csv(path, index_col='Unnamed: 0')

        return df

    def get_date(self):
        return self.date_list.copy()

    def get_path(self):
        return self.path

    def get_category(self):
        return self.category_list.copy()

    def get_user_num(self):
        return self.user_num


class combine:
    # date input은 항상 datetime object
    def __init__(self, user_num, interval=1):
        self.path = dataset_path + f'/combine/interval_{interval}/user{user_num}'
        self.user_num = user_num

        self.date_list = list(
            map(datetime.fromtimestamp,
                map(int, [file_name.split('.')[0] for file_name in os.listdir(self.path)])
                # path를 받아오는 거니까 string을 int로 mapping, .csv file 이니까 .csv 떼주기
                )
        )
        self.date_list.sort()

        self.label_col_list = get_label_col_list()

    # s_day: start day | e_day: end day
    # 특정 datetime의 첫번째 index timestamp부터 마지막 index timestamp까지
    def get_combine(self, target_col_list: list):
        if type(target_col_list) is not list:
            target_col_list = [target_col_list]

        df = pd.DataFrame()

        for day in self.date_list:
            df = df.append(
                self.get_combine_data(day)  # 이건 아래에 있는 function
            )

        drop_col = list(set(get_label_col_list()) - set(target_col_list))

        df = df.drop(drop_col, axis=1)
        df = df.reset_index(drop=True)
        return df

    def get_combine_data(self, day: datetime, interval: int = 1):
        day_ts = mk_ts(day)
        path = self.path + f'/{day_ts}.csv'

        if not os.path.exists(path):
            ll = lifelog(self.user_num)
            cat_list = ll.get_category()
            df = get_sensor_label(ll, day, cat_list, interval)
            df.to_csv(path)
        else:
            df = pd.read_csv(path, index_col='Unnamed: 0')

        return df

    def get_date(self):
        return self.date_list.copy()

    def get_path(self):
        return self.path

    def get_label_col(self):
        return self.label_col_list.copy()

    def get_user_num(self):
        return self.user_num
