import pandas as pd
from functools import reduce


def resume_process(file_name):
    # file_name = 'raw_data/stu_base.csv'

    apply_file = 'processed_data/apply.csv'
    english_file = 'processed_data/english.csv'
    grades_file = 'processed_data/grades.csv'
    honour_file = 'processed_data/honour.csv'
    project_file = 'processed_data/project.csv'
    ranking_file = 'processed_data/ranking.csv'
    university_file = 'processed_data/university.csv'
    label_file = 'processed_data/label.csv'
    score_file = 'processed_data/score.csv'

    # file = pd.read_csv(file_name, encoding='utf-8')
    apply = pd.read_csv(apply_file, encoding='utf-8')
    english = pd.read_csv(english_file, encoding='utf-8')
    grades = pd.read_csv(grades_file, encoding='utf-8')
    honour = pd.read_csv(honour_file, encoding='utf-8')
    project = pd.read_csv(project_file, encoding='utf-8')
    ranking = pd.read_csv(ranking_file, encoding='utf-8')
    university = pd.read_csv(university_file, encoding='utf-8')
    label = pd.read_csv(label_file, encoding='utf-8')
    score = pd.read_csv(score_file, encoding='gbk')

    # processed_file = pd.DataFrame()

    dfs = [apply, english, grades, honour, project, ranking, university, label]

    processed_file = reduce(lambda left, right: pd.merge(left, right, on="uid"), dfs)

    import ipdb
    ipdb.set_trace()
    processed_file = pd.merge(processed_file, score, left_on='uid', right_on='uid', how='outer')

    keep_list = ["uid", "apply_type", 'cet4','cet6','grade_points','honour','project','ranking_absolu','ranking','level','SSDM','label', 'score']

    processed_file = processed_file[keep_list]

    processed_file = processed_file.fillna(0)

    processed_file = processed_file[processed_file["grade_points"] > 0]
    processed_file = processed_file[processed_file["grade_points"] < 1]

    processed_file.columns = processed_file.columns.str.replace('total', 'score')
    processed_file.columns = processed_file.columns.str.replace('SSDM', 'location')

    processed_file = processed_file[processed_file["ranking_absolu"] > 1]
    processed_file.to_csv("processed_file.csv")


if __name__ == '__main__':
    resume_process("stu_base.csv")
