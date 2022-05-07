import pandas as pd
import os
from Lifelog import lifelog

path = r'../dataset/lifelog/2020/'
integ_path = r'../dataset/integ'


def create_dir(path: str):
    try:
        if not os.path.exists(path):
            create_dir(os.path.split(path)[0])
            os.makedirs(path)
    except OSError:
        print('Error: Creating directory. ' + path)


def integ(userNum: str):
    mk_ts = lambda d: int(d.timestamp())

    create_dir(path)
    create_dir(integ_path)

    try:
        ll = lifelog(userNum)
    except:
        return

    date_list = ll.get_date()
    category_list = ll.get_category()
    category_list.append('label')

    for i, day in enumerate(date_list):
        day_ts = mk_ts(day)
        res_path = f'{integ_path}/user{userNum}/{day_ts}'
        create_dir(res_path)
        print(f'----date: {day} ({i+1}/{len(date_list)})')
        for category in category_list:
            print(f'  --category: {category}')
            file_path = f'{res_path}/{day_ts}_{category}.csv'
            if not os.path.exists(file_path):
                df = ll.get_sensor(category=category, s_day=day)
                df.to_csv(file_path)
            else:
                print(f'    {file_path} is already exist')


if __name__ == "__main__":
    l = [str(i) for i in range(1, 31)]

    for n in l:
        integ(n)
