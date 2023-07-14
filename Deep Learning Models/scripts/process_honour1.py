# encoding = gbk
import pandas as pd


def string_concat(x):
    result = ''
    for i in x.values:
        result += i
    return result


def process_honour(file_name, honour_name):
    # file_name = 'raw_data/stu_base.csv'

    file = pd.read_csv(file_name, encoding='gbk')

    honour = pd.read_csv(honour_name, encoding='gbk')

    file.columns = [x[1:-1] for x in file.columns]

    file = pd.merge(file, honour, left_on="i", right_on='uid', how='outer')

    file["honour_name"] = file["honour_name"].apply(str)
    file["specificdesc"] = file["specificdesc"].apply(str)
    file["certificate_level"] = file["certificate_level"].apply(str)
    file["honour"] = file.honour_name.str.cat(file.specificdesc.str.cat(file.certificate_level))

    # drop_list = ['gender', 'political_status', 'nationality', 'idcard',
    #        'graduate_university', 'graduate_college', 'graduate_subject',
    #        'graduate_professional_class', 'graduate_profession',
    #        'target_university', 'target_college', 'target_subject',
    #        'target_profession', 'target_professor', 'target_professor2',
    #        'target_professor3', 'value_cet4', 'value_cet6', 'pro_stu_num',
    #        'grade_point', 'ranking', 'universitylevel', 'relation', 'remarks',
    #        'apply_type', 'examid', 'phone', 'foreign_language', 'gre_score',
    #        'toefl_score', 'email', 'user_name', 'nnamed: 3', 'honour_id', 'uid',
    #        'honour_name', 'specificdesc', 'certificate', 'certificate_level',
    #        'honour_at', 'create_at']
    # for drop_name in drop_list:
    #     file = file.drop(drop_name, axis=1)

    new_honour = file[["i", "honour"]]
    new_honour.columns = ["uid", "honour"]

    honour_group = new_honour.groupby("uid")
    honour = honour_group.apply(string_concat)
    new_honour = pd.DataFrame({"uid": [x for x in honour.index], "honour": [x[0] for x in honour.values]})

    new_honour.to_csv("processed_data/honour.csv")


if __name__ == '__main__':
    process_honour('stu_base.csv', 'stu_honour.csv')
