import pandas as pd


def apply2number(x):
    if x == '"putong"':
        return 1
    else:
        return 2


def process_apply(file_name):

    file = pd.read_csv(file_name, encoding='gbk')

    file.columns = [x[1:-1] for x in file.columns]

    # print(file["apply_type"].head())

    apply = pd.DataFrame()

    apply["uid"] = file["i"]
    apply["apply_type"] = file["apply_type"].apply(apply2number)

    apply.to_csv("processed_data/apply.csv")
