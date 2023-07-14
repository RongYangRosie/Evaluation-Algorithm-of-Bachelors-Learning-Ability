import os.path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from datasets.utils import project2vector, honour2vector
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()


class Student_data(Dataset):
    def __init__(self, mode, root):
        self.mode = mode
        self.root = root

        self.list_id = []
        self.list_honour = []
        self.list_apply = []
        self.list_grades = []
        self.list_project = []
        self.list_ranking = []
        self.list_ranking_absolu = []
        self.list_universitylevel = []
        self.list_universitylocation = []
        self.list_label = []
        self.list_score = []
        self.list_cet4 = []
        self.list_cet6 = []

        honour_file = pd.DataFrame()
        project_file = pd.DataFrame()

        bert_pre_tokenizer = os.path.join(root, 'bert-base-chinese-vocab.txt')

        if self.mode == "train":
            file = os.path.join(root, 'train.csv')
        else:
            file = os.path.join(root, 'test.csv')

        df = pd.read_csv(file, encoding='utf-8')

        honour_file['honour'] = df['honour']
        project_file['project'] = df['project']
        honour_processed = honour2vector(honour_file, bert_pre_tokenizer)
        project_processed = project2vector(project_file, bert_pre_tokenizer)

        length = len(df["uid"])
        for i in range(length):
            self.list_id.append(df["uid"][i])
            self.list_honour.append(honour_processed[i])
            self.list_apply.append(df["apply_type"][i])
            self.list_grades.append(df["grade_points"][i])
            self.list_project.append(project_processed[i])
            self.list_ranking.append(df["ranking"][i])
            self.list_ranking_absolu.append(df["ranking_absolu"][i])
            self.list_universitylevel.append(df["level"][i])
            self.list_universitylocation.append(df["location"][i])
            self.list_label.append(df["label"][i])
            self.list_score.append(df["score"][i])
            self.list_cet4.append(df["cet4"][i])
            self.list_cet6.append(df["cet6"][i])

        self.length = len(self.list_id)

    def __getitem__(self, item):

        universitylocation = np.expand_dims([np.asarray(self.list_universitylocation[item])], 0)
        universitylevel = np.expand_dims([np.asarray(self.list_universitylevel[item])], 0)
        grades = np.expand_dims([np.asarray(self.list_grades[item])], 0)
        apply = np.expand_dims([np.asarray(self.list_apply[item])], 0)
        ranking_absolu = np.expand_dims([np.asarray(self.list_ranking_absolu[item])], 0)
        ranking = np.expand_dims([np.asarray(self.list_ranking[item])], 0)
        label = np.expand_dims([np.asarray(self.list_label[item])], 0)
        score = np.expand_dims([np.asarray(self.list_score[item])], 0)
        cet4 = np.expand_dims([np.asarray(self.list_cet4[item])], 0)
        cet6 = np.expand_dims([np.asarray(self.list_cet6[item])], 0)
        honour = np.asarray(self.list_honour[item])
        project = np.asarray(self.list_project[item])

        return universitylocation.astype(np.float32), \
               universitylevel.astype(np.float32), \
               grades.astype(np.float32), \
               apply.astype(np.float32), \
               ranking.astype(np.float32), \
               ranking_absolu.astype(np.float32), \
               label.astype(np.float32), \
               score.astype(np.float32), \
               cet4.astype(np.float32), \
               cet6.astype(np.float32), \
               honour, \
               project

    def __len__(self):
        return self.length


if __name__ == '__main__':
    student = Student_data(mode='train', root='')
    dataloader = torch.utils.data.DataLoader(student, batch_size=1, shuffle=False, num_workers=10)

    location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project = student[0]

    import ipdb

    ipdb.set_trace()

    print(location.shape)

# keras -> numpy 操作
# train和test数据集的划分
# project维度改成1024
# 将apply普通申请改为1，特殊申请改为2
