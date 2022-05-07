import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from workspace.util.User import lifelog, mk_ts
from workspace.util.Encoding import user_label_encoder


def check_running_time(message: str, st_time: time.time) -> time.time:
    print(f'{message}, Running time: {time.time() - st_time}')
    return time.time()


# for multi-processing
def parallel_search_interval_sensor_df(df: pd.DataFrame, start_ts: float, end_ts: float,
                                       interval: float) -> pd.DataFrame:
    ts_col = 'timestamp'
    num_cores = cpu_count()
    start_ts_list = list(map(lambda x: x - x % interval,
                             [start_ts + (end_ts - start_ts) / num_cores * i for i in range(0, num_cores)]))
    end_ts_list = list(map(lambda x: x - x % interval,
                           [start_ts + (end_ts - start_ts) / num_cores * i for i in range(1, num_cores + 1)]))

    interval_list = [interval] * num_cores  # starmap method에 non-iterable object를 넘겨주는 방법을 몰라서 그냥 iterable로 만들어서 넘김
    df_split = np.array_split(df, num_cores)
    with Pool() as pool:
        cat_df = pd.concat(
            pool.starmap(get_cat_df,
                         zip(df_split, start_ts_list, end_ts_list, interval_list)
                         ), axis=0).groupby(level=0).first()

    return cat_df


def get_cat_df(df: pd.DataFrame, start_ts: float, end_ts: float, interval: float) -> pd.DataFrame:
    # st_time = time.time()

    ts_col = 'timestamp'

    cur_ts = start_ts
    cat_df = pd.DataFrame()

    while cur_ts <= end_ts:
        interval_df = df[(cur_ts <= df[ts_col]) & (df[
                                                       ts_col] < cur_ts + interval)]  # 정해진 interval 사이의 timestamp 데이터프레임 (다른 컬럼도 출력됨(x,y,z 같은거) 헷갈리지 마셈)

        interval_mean = interval_df.mean()  # 그 간격 평균
        interval_mean[ts_col] = cur_ts  # 합친 row의 timestamp는 cur_ts

        cat_df = cat_df.append(interval_mean, ignore_index=True)
        cur_ts += interval

    cat_df = cat_df.set_index(ts_col, drop=True)

    # st_time = check_running_time(f'PID: {os.getpid()}', st_time)

    return cat_df


def get_sensor_combine_df(ll: lifelog, day: datetime, cat_list: list, interval: float,
                          sample_rate: float) -> pd.DataFrame:
    st_time = time.time()

    ts_col = 'timestamp'
    df_list = []

    minStart_ts = float('inf')
    maxEnd_ts = float('-inf')

    # for loop
    # description: 컬럼명 바꾸기, df에서 minStart_ts, maxStart_ts 구하기
    for categ in cat_list:
        df = ll.get_sensor(categ, day)

        df = df.iloc[:int(df.shape[0] * sample_rate), :]

        if df.shape[0] == 0:
            continue

        # category name prefix at columns except timestamp
        col_list = list(df.columns)
        new_col_list = [categ + '_' + col if col != ts_col else col for col in
                        col_list]  # timestamp일 땐 넘어가고 x,y,z이면 센서이름_x,y,z
        df.columns = new_col_list

        # append df at df_list
        df_list.append(df)
        start_ts = df.iloc[0, :][ts_col]  # 0번째 인덱스 timestamp
        end_ts = df.iloc[-1, :][ts_col]  # 마지막 인덱스 timestamp

        if start_ts < minStart_ts:
            minStart_ts = start_ts
        if end_ts > maxEnd_ts:
            maxEnd_ts = end_ts

    st_time = check_running_time(f'get dataframe', st_time)

    combine_df = pd.DataFrame()

    # for loop
    # timestamp 간격 interval 간격으로 바꿔서(일정하게 통일) cat_df에 할당
    # combine_df cat_df을 axis=1로 받기 (이 셀에서 최종 데이터프레임)
    for i, df in enumerate(df_list):
        cur_ts = minStart_ts
        cat_df = pd.DataFrame()

        cat_df = parallel_search_interval_sensor_df(df, minStart_ts, maxEnd_ts, interval)

        st_time = check_running_time(f'make {cat_list[i]}', st_time)

        combine_df = pd.concat([combine_df, cat_df], axis=1)  # cat_df 받기

    combine_df = combine_df.reset_index()
    # end of for loop

    return combine_df


def get_sensor_label_df(ll: lifelog, day: datetime, sensor_df: pd.DataFrame, target_col: list) -> pd.DataFrame:
    st_time = time.time()

    label_col = ['ts']
    label_col.extend(target_col)

    label_df = ll.get_sensor('label', day).loc[:, label_col]

    label_indice = list(label_df.index)
    pre_i = label_indice[0]

    sensor_label_df = pd.DataFrame()
    for i in label_indice[1:]:
        sensor_block_df = sensor_df[(label_df.loc[pre_i, 'ts'] <= sensor_df['timestamp']) & (
                sensor_df['timestamp'] < label_df.loc[i, 'ts'])].copy()
        sensor_block_df[target_col] = label_df.loc[pre_i, target_col]
        pre_i = i
        sensor_label_df = sensor_label_df.append(sensor_block_df)

    st_time = check_running_time('make sensor_label', st_time)

    return sensor_label_df


def get_sensor_label(ll: lifelog, day: datetime, category: list, interval: float, target_col: list,
                     sample_rate: float = 1) -> pd.DataFrame:
    st_time = time.time()

    print(f'get_sensor_label, user{ll.get_user_num()}, {mk_ts(day)}')

    if type(category) is not list:
        category = [category]
    if type(target_col) is not list:
        target_col = [target_col]

    sensor_df = get_sensor_combine_df(ll=ll, day=day, cat_list=category, interval=interval, sample_rate=sample_rate)
    sensor_label_df = get_sensor_label_df(ll=ll, day=day, sensor_df=sensor_df, target_col=target_col)

    st_time = check_running_time(f'get_sensor_label', st_time)

    return sensor_label_df


def encode_df(df: pd.DataFrame, target_col: list) -> pd.DataFrame:
    ul = user_label_encoder()
    if type(target_col) is not list:
        target_col = [target_col]

    df_encoded = df.copy()

    if type(df) is pd.Series:
        df_encoded = df.to_frame()

    for col in target_col:
        code_df = ul.get_label_code(col)
        le = LabelEncoder()
        le.fit(code_df.loc[:, col])

        try:
            df_encoded.loc[:, col] = le.transform(df_encoded.loc[:, col])
        except:
            print(f'----------{col} has error------------')
            raise

    return df_encoded
