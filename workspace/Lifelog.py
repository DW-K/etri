import pandas as pd
import os
from datetime import datetime
import time
from bisect import bisect

dataset_path = r"../dataset/"
# pd.options.display.float_format = '{:.20f}'.format


class lifelog:
    # date input은 항상 datetime object
    def __init__(self, user_num):
      self.path = dataset_path + f'lifelog/2020/user{user_num}/'

      self.date_list = list(
          map(datetime.fromtimestamp,
              map(int, os.listdir(self.path))
              )
      )

      self.category_list = ['e4Acc', 'e4Bvp', 'e4Eda', 'e4Hr',
                            'e4Temp', 'mAcc', 'mGps', 'mGyr', 'mMag']

    def get_sensor(self, category: str, s_day: datetime, e_day: datetime = None):
        if e_day is None:
            e_day = s_day

        date_start_index = bisect(self.date_list, s_day)
        date_end_index = bisect(self.date_list, e_day)

        df = pd.DataFrame()

        if category == 'label':
            for i in range(date_start_index-1, date_end_index):
                if df.shape[0] > 0:
                    df = df.append(
                        self.get_label_data(self.date_list[i])
                    )
                else:
                    df = self.get_label_data(self.date_list[i])
            return df

        elif category in self.category_list:
            for i in range(date_start_index-1, date_end_index):
                ts_list = self.get_ts(category, self.date_list[i])
                for ts in ts_list:
                    if df.shape[0] > 0:
                        df = df.append(
                            self.get_sensor_data(category, self.date_list[i], ts)
                        )
                    else:
                        df = self.get_sensor_data(category, self.date_list[i], ts)

            df = df.reset_index(drop=True)
            return df

        else:
            print('wrong category')

    def get_label_data(self, day: datetime):
        mk_ts = lambda d: int(d.timestamp())
        path = self.path + f'{mk_ts(day)}/{mk_ts(day)}' + '_label.csv'
        df = pd.read_csv(path)

        return df

    def get_ts(self, category: str, day: datetime):
        mk_ts = lambda d: int(d.timestamp())
        path = self.path + f'{mk_ts(day)}' + f'/{category}'
        ts_list = list(
            map(datetime.fromtimestamp,
                map(int,
                    [n.split('.')[0] for n in os.listdir(path)]
                )
            )
        )
        ts_list.sort()
        return ts_list

    def get_sensor_data(self, category: str, day: datetime, ts: datetime):
        mk_ts = lambda d: int(d.timestamp())
        ts = mk_ts(ts)
        path = self.path + f'{mk_ts(day)}' + f'/{category}' + f'/{ts}' + '.csv'

        df = pd.read_csv(path)

        df['timestamp'] += ts

        return df

    def get_date(self):
        return self.date_list.copy()

    def get_path(self):
      return self.path

    def get_category(self):
        return self.category_list.copy()
