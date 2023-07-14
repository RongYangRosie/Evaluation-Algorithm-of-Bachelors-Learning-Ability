import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from pytorch_pretrained_bert import BertModel

from datasets.dataset import Student_data


class Classification(nn.Module):
    def __init__(self, usegpu=True):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(1542, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bert = BertModel.from_pretrained('bert-base-chinese').eval()

    def forward(self, location, level, grades, apply, ranking, ranking_absolu, honour, project):
        with torch.no_grad():
            honour = self.bert(honour)[1].unsqueeze(-1)
            project = self.bert(project[:, :512])[1].unsqueeze(-1)
        features = torch.cat([location, level, grades, apply, ranking, ranking_absolu, honour, project],
                             dim=1)
        x = features.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


if __name__ == '__main__':
    student = Student_data(mode='train', root='')
    dataloader = torch.utils.data.DataLoader(student, batch_size=10, shuffle=True, num_workers=1)

    test_student = Student_data(mode='test', root='')
    test_dataloader = torch.utils.data.DataLoader(test_student, batch_size=10, shuffle=False, num_workers=1)

    model = Classification().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in tqdm.tqdm((range(100))):
        loss_list = []
        for i, data in enumerate(dataloader):
            data = [d.cuda() for d in data]
            location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project, cet4, cet6 = data

            out = model(location, level, grades, apply, ranking, ranking_absolu, honour, project)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(out, label.squeeze(-1))
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
                location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project = data

                out = model(location, level, grades, apply, ranking, ranking_absolu, honour, project)
                label = label.squeeze(-1)
                loss = F.binary_cross_entropy(out, label)
                loss_list.append(loss.detach().item())

                correct += ((out > 0.5).long() == label.long()).sum()
                # print(out, label)
                total += len(out)
            acc = correct.item() / total
            print("test loss: {0}".format(np.mean(loss_list)))
            print("Accuracy : {0:.2f}".format(acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "epoch{0}, accuracy{1:.2f}.pth".format(epoch, correct.item() / total))

# BN
# dropout
# 数据增强
# 调整学习率
