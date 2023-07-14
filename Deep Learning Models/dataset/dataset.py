import torch.nn as nn
import torch
import numpy as np
import pickle
import torch.utils.data as data
from torch.utils.data import Dataset
import os.path
import pandas as pd
from utils import project2vector, honour2vector


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
            self.list_honour.append(honour_processed["honour"][i])
            self.list_apply.append(df["apply_type"][i])
            self.list_grades.append(df["grade_points"][i])
            self.list_project.append(project_processed["project"][i])
            self.list_ranking.append(df["ranking"][i])
            self.list_ranking_absolu.append(df["ranking_absolu"][i])
            self.list_universitylevel.append(df["level"][i])
            self.list_universitylocation.append(df["location"][i])
            self.list_label.append(df["label"][i])
            self.list_score.append(df["score"][i])

        self.length = len(self.list_id)

    def __getitem__(self, item):

        universitylocation = np.array(self.list_universitylocation[item])
        universitylevel = np.array(self.list_universitylevel[item])
        grades = np.array(self.list_grades[item])
        apply = np.array(self.list_apply[item])
        ranking_absolu = np.array(self.list_ranking_absolu[item])
        ranking = np.array(self.list_ranking[item])
        label = self.list_label[item]
        score = np.array(self.list_score[item])

        return torch.from_numpy(universitylocation.astype(np.float32)),\
               torch.from_numpy(universitylevel.astype(np.float32)),\
               torch.from_numpy(grades.astype(np.float32)),\
               torch.from_numpy(apply.astype(np.float32)),\
               torch.from_numpy(ranking.astype(np.float32)),\
               torch.from_numpy(ranking_absolu.astype(np.float32)), \
               torch.LongTensor(label.astype(np.int32)),\
               torch.from_numpy(score.astype(np.float32))

    def __len__(self):
        return self.length


if __name__ == '__main__':
    student = Student_data(mode='train', root='')
    dataloader = torch.utils.data.DataLoader(student, batch_size=1, shuffle=True, num_workers=10)
    for i, data in dataloader:
        if i == 0:
            print(data)
        else:
            break
