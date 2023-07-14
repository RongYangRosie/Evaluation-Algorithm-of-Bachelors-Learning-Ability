import torch
import visdom
vis = visdom.Visdom(env='test1')
x = torch.arange(1,30,0.01)
y = torch.sin(x)
vis.line(X=x,Y=y,win='sinx',opts={'title':'y=sin(x)'})
