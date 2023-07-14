import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import visdom
from pytorch_pretrained_bert import BertModel

from datasets.dataset import Student_data

GLOBAL_SEED = 1
GLOBAL_WORKER_ID = None
vis1 = visdom.Visdom(env='bert_train')
vis2 = visdom.Visdom(env='bert_test')
vis3 = visdom.Visdom(env='bert_acc')
vis1.line([[0., 0.]], [0], win='train', opts=dict(title='training_loss'))
vis2.line([[0., 0.]], [0], win='test', opts=dict(title='test'))
vis3.line([[0., 0.]], [0], win='acc', opts=dict(title='acc'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(work_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = work_id
    set_seed(GLOBAL_WORKER_ID + work_id)


class Classification(nn.Module):
    def __init__(self, usegpu=True):
        super(Classification, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(1544, 256),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.bert = BertModel.from_pretrained('bert-base-chinese').eval()

    def forward(self, location, level, grades, apply, ranking, ranking_absolu, honour, project, cet4, cet6):
        with torch.no_grad():
            honour = self.bert(honour)[1]
            project = self.bert(project[:, :512])[1]
        # honour = self.fc1(honour)
        # project = self.fc1(project)
        features = torch.cat([location, level, grades, apply, ranking, ranking_absolu, honour.unsqueeze(-1), project.unsqueeze(-1), cet4, cet6],
                             dim=1)
        x = features.squeeze(-1)
        x = torch.sigmoid(self.fc1(x))

        return x


if __name__ == '__main__':
    student = Student_data(mode='train', root='')
    dataloader = torch.utils.data.DataLoader(student, batch_size=10, shuffle=True, num_workers=1, drop_last=True,
                                             worker_init_fn=worker_init_fn)

    test_student = Student_data(mode='test', root='')
    test_dataloader = torch.utils.data.DataLoader(test_student, batch_size=10, shuffle=False, num_workers=1)

    model = Classification().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    step_train = []
    train_loss = []
    step_test = []
    test_loss = []
    for epoch in tqdm.tqdm((range(100))):
        loss_list = []
        for i, data in enumerate(dataloader):
            data = [d.cuda() for d in data]
            location, level, grades, apply, ranking, ranking_absolu, label, score,cet4, cet6, honour, project = data

            out = model(location, level, grades, apply, ranking, ranking_absolu, honour, project, cet4, cet6)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(out, label.squeeze(-1))
            train_loss.append(loss.detach().item())
            n_iter = epoch * len(dataloader) + i
            step_train.append(n_iter)
            vis1.line([loss.item()], [n_iter], win='train', update='append')
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().item())
        print("train loss:{0}".format(np.mean(loss_list)))

        model.eval()

        correct = 0
        total = 0
        best_acc = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                data = [d.cuda() for d in data]
                location, level, grades, apply, ranking, ranking_absolu, label, score, cet4, cet6, honour, project = data

                out = model(location, level, grades, apply, ranking, ranking_absolu, honour, project, cet4, cet6)
                label = label.squeeze(-1)
                loss = F.binary_cross_entropy(out, label)
                test_loss.append(loss.detach().item())
                n_iter = epoch * len(test_dataloader) + i
                step_test.append(n_iter)
                loss_list.append(loss.detach().item())

                correct += ((out > 0.5).long() == label.long()).sum()
                # print(out, label)
                total += len(out)
                vis2.line([loss.item()], [n_iter],win='test', update='append')
            acc = correct.item() / total
            vis3.line([acc], [epoch], win='acc', update='append')
            print("test loss: {0}".format(np.mean(loss_list)))
            print("Accuracy : {0:.2f}".format(acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "epoch{0}, accuracy{1:.2f}.pth".format(epoch, correct.item() / total))
# BN
# dropout
# 数据增强
# 调整学习率
