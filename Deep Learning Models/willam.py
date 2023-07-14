import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from datasets.dataset import Student_data
from progress.bar import Bar
import visdom
vis1 = visdom.Visdom(env='bert_train')
vis2 = visdom.Visdom(env='bert_test')
vis3 = visdom.Visdom(env='bert_acc')
vis1.line([[0., 0.]], [0], win='train', opts=dict(title='training_loss'))
vis2.line([[0., 0.]], [0], win='test', opts=dict(title='test'))
vis3.line([[0., 0.]], [0], win='acc', opts=dict(title='acc'))


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(768, 1, bias=True)
        self.fc2 = nn.Linear(768, 1, bias=True)
        self.fc3 = nn.Linear(8, 1, bias=True)
        self.relu = torch.nn.ReLU()

        torch.nn.init.uniform_(self.fc1.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.fc2.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.fc3.weight, a=-0.2, b=0.2)

    def forward(self, location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project):
        honour = torch.sigmoid(self.fc1(honour))
        project = torch.sigmoid(self.fc2(project))
        features = torch.cat((honour, project, location, level, grades, apply, ranking, ranking_absolu), dim=1)

        x = F.sigmoid(self.fc3(features))

        return x


class network2(nn.Module):
    def __init__(self):
        super(network2, self).__init__()
        self.fc = nn.Linear(1542, 1, bias=True)
        torch.nn.init.uniform_(self.fc.weight, a=-0.2, b=0.2)

    def forward(self, location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project):
        features = torch.cat((honour, project, location, level, grades, apply, ranking, ranking_absolu), dim=1)

        return torch.sigmoid(self.fc(features))


class network3(nn.Module):
    def __init__(self):
        super(network3, self).__init__()
        self.fc1 = nn.Linear(768, 16, bias=True)
        self.fc2 = nn.Linear(768, 16, bias=True)
        self.fc3 = nn.Linear(38, 1, bias=True)

        torch.nn.init.uniform_(self.fc1.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.fc2.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.fc3.weight, a=-0.2, b=0.2)

    def forward(self, location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project):
        honour = torch.sigmoid(self.fc1(honour))
        project = torch.sigmoid(self.fc2(project))
        features = torch.cat((honour, project, location, level, grades, apply, ranking, ranking_absolu), dim=1)

        x = F.sigmoid(self.fc3(features))

        return x


class network4(nn.Module):
    def __init__(self):
        super(network4, self).__init__()
        self.fc1 = nn.Linear(768, 16, bias=True)
        self.fc2 = nn.Linear(768, 16, bias=True)
        self.fc3 = nn.Linear(38, 1, bias=True)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(16)

        torch.nn.init.uniform_(self.fc1.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.fc2.weight, a=-0.2, b=0.2)
        torch.nn.init.uniform_(self.fc3.weight, a=-0.2, b=0.2)

    def forward(self, location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project):
        honour = self.relu(self.bn1(self.fc1(honour)))
        project = self.relu(self.bn2(self.fc2(project)))
        features = torch.cat((honour, project, location, level, grades, apply, ranking, ranking_absolu), dim=1)

        x = F.sigmoid(self.fc3(features))

        return x


def evaluate(model, thres, output_diff=False):
    model.eval()
    student = Student_data(mode='test', root='')
    dataloader = torch.utils.data.DataLoader(student, batch_size=1, shuffle=False, num_workers=1)

    error = 0
    for i, batch in enumerate(dataloader):
        location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project =\
            batch
        location, level, grades, apply, ranking, ranking_absolu, score =\
            location.cuda(), level.cuda(), grades.cuda(), \
            apply.cuda(), ranking.cuda(), ranking_absolu.cuda(),\
            score.cuda()
        n_iter1 = e * len(dataloader) + i
        honour = student.bc.encode(list(honour)).copy()
        honour = torch.from_numpy(honour).cuda()
        project = student.bc.encode(list(project)).copy()
        project = torch.from_numpy(project).cuda()

        out = model(
            location, level, grades, apply, ranking,
            ranking_absolu, label, score, honour, project
        ).cpu().detach().numpy()

        if out.item() > thres:
            out_ = 1
        else:
            out_ = 0

        label_ = label.numpy().item()
        if abs(label_ - out_) > 1e-3:
            error += 1
        else:
            if out_ >= thres and output_diff:
                print(location, level, grades, apply, ranking,
                      ranking_absolu, label, score, honour, project)
        vis2.line([error.item()], [n_iter1], win='test', update='append')
    return error / len(student)


if __name__ == '__main__':
    epoch_num = 100
    batch_size = 64

    student = Student_data(mode='train', root='')
    dataloader = torch.utils.data.DataLoader(student, batch_size=batch_size, shuffle=True, num_workers=1)
    model = Classification()
    model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6
    )

    total_iterations = epoch_num * (len(student) // batch_size + 1)
    bar = Bar(max=total_iterations)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_iterations,
        eta_min=1e-6
    )

    loss = nn.BCELoss()

    min_error = 1
    for e in range(epoch_num):
        model.train()
        for i, batch in enumerate(dataloader):
            location, level, grades, apply, ranking, ranking_absolu, label, score, honour, project =\
                batch
            location, level, grades, apply, ranking, ranking_absolu, label, score =\
                location.cuda(), level.cuda(), grades.cuda(), \
                apply.cuda(), ranking.cuda(), ranking_absolu.cuda(),\
                label.cuda(), score.cuda()

            honour = torch.from_numpy(student.bc.encode(list(honour))).cuda()
            project = torch.from_numpy(student.bc.encode(list(project))).cuda()

            out = model(
                location, level, grades, apply, ranking,
                ranking_absolu, label, score, honour, project
            )
            n_iter = e * len(dataloader) + i
            cost = loss(out, label)
            vis1.line([cost.item()], [n_iter], win='train', update='append')
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            scheduler.step()
            bar.next()

        error = evaluate(model, 0.5, output_diff=(e>8))

        vis3.line([error], [e], win='acc', update='append')
        print("epoch=%d error=%.2f" % (e, error))

        if error < min_error:
            torch.save(model, "checkpoint/network4/network_%.2f.pth" % error)
            print("save model when error=%.2f" % error)
            min_error = error

    bar.finish()