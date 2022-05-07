import datetime
import os
import traceback

import pandas as pd

from workspace.Lifelog import lifelog
from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list, get_category_list, mk_ts, get_total_date_list, dataset_path
from workspace.util.combine_dataset import get_sensor_label, encode_df


def make_dataset(day: datetime, interval: float):
    user_list = get_user_list()
    category = get_category_list()

    ul = user_label_encoder()
    target_col = ul.get_label()

    df_appended = pd.DataFrame()

    interval_path = dataset_path + f"/combine/interval_{interval}"
    if not os.path.exists(interval_path):
        os.mkdir(interval_path)

    for user in user_list:
        ll = lifelog(user)
        user_num = ll.get_user_num()
        day_ts = mk_ts(day)

        res_path = interval_path + f'/user{user_num}'
        file_path = res_path + f"/{day_ts}.csv"

        if not os.path.exists(res_path):  # 폴더 없으면 폴더 만들기
            os.mkdir(res_path)
        else:
            if os.path.exists(file_path):  # 이미 파일 있는지 체크
                # print(f'skip user{user_num}, {day_ts}')
                continue

        if day not in ll.get_date():
            continue

        # get data
        try:
            df = get_sensor_label(ll=ll, day=day, category=category, interval=interval, target_col=target_col,
                                  sample_rate=1)
        except Exception as e:
            print(f'user{user_num}: {traceback.format_exc()}')
            continue
        # print(f'need data user{ll.get_user_num()}, {day_ts}')

        if df.shape[0] == 0:
            continue

        # df_encoded = encode_df(df, target_col)

        print('-----------start write file-------------')
        df.to_csv(file_path)
        print(f'save {df.shape[0]} rows')
        print('-----------complete write file-----------')


if __name__ == "__main__":
    date_list = get_total_date_list()

    for i, day in enumerate(date_list):
        print(f'{i}/{len(date_list)}')
        make_dataset(day, 1)