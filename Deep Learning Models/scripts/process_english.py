import pandas as pd


def process_english(file_name):

    # file_name = 'raw_data/stu_base.csv'
    file = pd.read_csv(file_name, encoding='gbk')

    file.columns = [x[1:-1] for x in file.columns]

    # print(file["value_cet4"].head())
    # print(file["value_cet6"].head())

    english = pd.DataFrame()
    english["uid"] = file["i"]
    english["cet4"] = file["value_cet4"].apply(replace)
    english["cet6"] = file["value_cet6"].apply(replace)

    english.to_csv("processed_data/english.csv")


def replace(x):
    if x == '""':
        return 0
    else:
        return x
