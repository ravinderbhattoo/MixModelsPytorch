from MixModelsPytorch.mmodel import MModel
import sys; sys.path = [".."] + sys.path

import MixModelsPytorch
import torch
import torch.nn as nn
import numpy as np
from torch import optim

model = MixModelsPytorch.MModel((2, 1))
model.addmodel('nn', nn.Linear(2, 3))
model.addmodel('I', lambda x: x)
model.addmodel('ME', lambda X, s: X[:, 0]+X[:, 1]+X[:, 2]+s[:, 0])

def func(X1, X2, F_):
    x = F_.nn(X1)
    T = F_.I(X2)
    Visc = F_.ME(x, T)
    return Visc, x[:, 1] 

model.setforward(func)


def lossfn(X, y1, y2):
    v1, v2 = model(X)
    return ((v1-y1)**2 + (v2-y2)**2).mean()


X = torch.FloatTensor([[1, 2, 3, 4], [1, 2, 3, 4]])
y1 = torch.FloatTensor([2, 4])
y2 = torch.FloatTensor([2, 4])

optimizer = optim.SGD(model.parameters(), lr=0.01)
for i in range(100):
    optimizer.zero_grad()
    loss = lossfn(X, y1, y2)
    loss.backward()
    optimizer.step()
    if i%10==0:
        print(i, loss.item())

print(model(X))
model.save("mymodel.mmodel")