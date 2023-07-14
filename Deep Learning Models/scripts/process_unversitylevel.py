# encoding = gbk
import pandas as pd


def level985(x):
    if x == 1:
        return 7
    else:
        return 0


def level211(x):
    if x == 1:
        return 1
    else:
        return 0


def levelnormal(x):
    if x == 1:
        return 1
    else:
        return 0


def process_university(file_name, university_file):
    # file_name = 'raw_data/stu_base.csv'
    # university_file = 'raw_data/db_university_free.csv'

    file = pd.read_csv(file_name, encoding='gbk')
    db_file = pd.read_csv(university_file, encoding='gbk')

    file.columns = [x[1:-1] for x in file.columns]

    university = pd.DataFrame()
    university['uid'] = file["i"]
    university["level"] = file["universitylevel"]

    university['id'] = file["graduate_university"].apply(lambda x: x[1:-1])
    university['id'] = university['id'].apply(int)

    result = pd.merge(university, db_file, left_on="id", right_on="university_id", how="outer")

    drop_list = ['university_id', 'SSDMC', 'id', 'university_name', 'level985', 'level211', 'levelnormal', 'is985', 'is211', 'freetest_qualified']

    result = result.fillna(0)

    result = result[result["id"] > 0]

    # process 985, 211 schools
    result["level985"] = result["is985"].map(level985)

    result["level211"] = result["is211"].map(level211)

    result["levelnormal"] = result["freetest_qualified"].map(levelnormal)

    result["level"] = result.apply(lambda x: x["level985"] + x["level211"] + x["levelnormal"], axis=1)

    for drop_name in drop_list:
        result = result.drop(drop_name, axis=1)

    # print(result.head())

    result.to_csv("processed_data/university.csv")
