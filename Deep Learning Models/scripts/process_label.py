import pandas as pd


def process_label(text, label):
    file = pd.read_csv(text, encoding='gbk')
    label = pd.read_csv(label, encoding='utf-8')

    file.columns = [x[1:-1] for x in file.columns]

    result = pd.DataFrame()
    result["label"] = 0

    file["new_idcard"] = file["idcard"].apply(lambda x: x[1:-1])
    file['new_idcard'] = file['new_idcard'].apply(lambda x: x[:6] + x[-4:])

    label['new_idcard'] = label['身份证号'].apply(lambda x: x[:6] + x[-4:])

    result = pd.merge(file, label, left_on="new_idcard", right_on='new_idcard', how='outer')
    result["score"] = result["总分"]
    result.loc[result.score > 0, 'label'] = 1
    result = result[["i", "label"]]

    result.columns = ["uid", "label"]

    result.to_csv("processed_data/label.csv")


if __name__ == '__main__':
    process_label("stu_base.csv", "accept.csv")
