import os
import pandas

from utils.datacollection.tools import downstream_task_new,test_task_new
from task_mapping import task_dict, base_path



for path in os.listdir(f'{base_path}/.'):
    if not path.__contains__('hdf'):
        continue
    print(path)
    df = pandas.read_hdf(os.path.join(base_path, path))
    name = path.split('.')[0]
    task_type = task_dict[name]
    y = df.iloc[:, -1]
    if path.__contains__('bike'):
        print(df)
        x = df.iloc[:,:-4]
    else:
        x = df.iloc[:, :-1]
    print(downstream_task_new(df, task_type, None))
    print(test_task_new(df, task_type))


