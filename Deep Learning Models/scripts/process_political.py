import pandas as pd


def nationality_process(x):
    if x == '"01"' or x == '"02"':
        return 1
    else:
        return 0


file_name = 'raw_data/stu_base.csv'

file = pd.read_csv(file_name, encoding='gbk')

file.columns = [x[1:-1] for x in file.columns]

print(file["political_status"].head())

# file[[file["nationality"] == '"01"']] = 0
file["political_status"] = file["political_status"].apply(nationality_process)
print(file["political_status"])

nationality = file[["i", "political_status"]]

nationality.to_csv('processed_data/political.csv')
