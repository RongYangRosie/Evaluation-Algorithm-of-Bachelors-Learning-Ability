import pandas as pd


def nationality_process(x):
    if x == '"01"':
        return 0
    else:
        return 1


file_name = 'raw_data/stu_base.csv'

file = pd.read_csv(file_name, encoding='gbk')

file.columns = [x[1:-1] for x in file.columns]

print(file["nationality"].head())

# file[[file["nationality"] == '"01"']] = 0
file["nationality"] = file["nationality"].apply(nationality_process)
print(file["nationality"])

nationality = file[["i", "nationality"]]

nationality.to_csv('processed_data/nationality.csv')
