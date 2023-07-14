# encoding = gbk
import pandas as pd


def string_concat(x):
    result = ''
    for i in x.values:
        result += i
    return result


def process_project(file_name, project_name):
    # file_name = 'raw_data/stu_base.csv'

    # file = 'stu_base.csv'
    # project_name = 'stu_project.csv'
    df = pd.read_csv(file_name, encoding='gbk')
    project = pd.read_csv(project_name, encoding='gbk')
    # %%
    df.columns = [x[1:-1] for x in df.columns]
    result = pd.merge(df, project, left_on="i", right_on='uid', how='outer')
    result["project_name"] = result["project_name"].apply(str)
    result["achievement"] = result["achievement"].apply(str)
    result['certificate_level'] = result['certificate_level'].apply(str)
    result['project'] = result.project_name.str.cat(result.achievement.str.cat(result.certificate_level))
    # %%
    processed_project = result[["i", "project"]]
    processed_project.columns = ["uid", "project"]

    project_group = processed_project.groupby("uid")
    project = project_group.apply(string_concat)
    project2 = pd.DataFrame({"uid": [x for x in project.index], "project": [x[0] for x in project.values]})
    project2.to_csv("processed_data/project.csv")


if __name__ == '__main__':
    process_project('stu_base.csv', 'stu_project.csv')
