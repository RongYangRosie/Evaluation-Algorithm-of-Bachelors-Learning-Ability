import pandas as pd


def process_score(file_name):

    # file_name = 'raw_data/stu_base.csv'
    file = pd.read_csv(file_name, encoding='gbk')

    # file.columns = [x[0:-1] for x in file.columns]

    score = pd.DataFrame()
    score["uid"] = file["id"]
    score["score"] = file["total"]

    score.to_csv("processed_data/score.csv")
