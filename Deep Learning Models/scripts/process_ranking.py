import pandas as pd


def process_ranking(file_name):
    # file_name = 'raw_data/stu_base.csv'

    file = pd.read_csv(file_name, encoding='gbk')

    file.columns = [x[1:-1] for x in file.columns]

    ranking = pd.DataFrame()
    ranking["uid"] = file["i"]
    ranking["ranking_absolu"] = 1 / (file["ranking"] / file["pro_stu_num"])
    ranking["ranking"] = file["ranking"]

    ranking.to_csv("processed_data/ranking.csv")