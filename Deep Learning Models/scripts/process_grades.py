import pandas as pd


def grades_normalize(x):
    if x > 10:
        return x / 100
    elif 10 > x > 5:
        return x / 10
    elif x > 4.0:
        return x / 5.0
    else:
        return x / 4.0


def process_grades(file_name):
    # file_name = 'raw_data/stu_base.csv'
    file = pd.read_csv(file_name, encoding='gbk')

    file.columns = [x[1:-1] for x in file.columns]

    grades = pd.DataFrame()

    grades["uid"] = file["i"]
    grades["grade_points"] = file["grade_point"].apply(grades_normalize)

    grades.to_csv("processed_data/grades.csv")
