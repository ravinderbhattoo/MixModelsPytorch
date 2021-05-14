from MixModelsPytorch.mmodel import create_sequential_model
from MixModelsPytorch import MModel, load, create_sequential_model
import os

import torch
import torch.nn as nn
from torch import optim
# torch.autograd.set_detect_anomaly(True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_csv("./data/Visc_At_Tliq_SciGlass_1.csv")

NAME = "mlp_with_myega"
os.makedirs(f"{NAME}", exist_ok=True)

X = torch.FloatTensor(data.values[:, :-1])
VISC = torch.FloatTensor(data.values[:, -1:])

N = X.shape[1]

model = MModel((N-1, 1))
NN = create_sequential_model([N-1, 30, 20, 3])  
log = torch.log10 # not need to use this
def MYEGA(n, tg, m, t):
    return n + tg/t*(12 - n)*torch.exp((tg/t-1)*(m/(12-n)-1))


model.addmodel('nn', NN)
model.addmodel('I', lambda x: x)
model.addmodel('ME', MYEGA)

def func(X1, X2, F_):
    x = F_.nn(X1)
    T = F_.I(X2)
    Visc = F_.ME(x[:,0:1], x[:,1:2], x[:,2:3], T[:,0:1])
    return Visc

model.setforward(func)

def lossfn(X, y):
    visc_ = model(X)
    return ((visc_-y)**2).mean()

optimizer = optim.SGD(model.parameters(), lr=1.0e-3)
optimizer.zero_grad()

epochs = 10000
LOSS =  []
for i in range(epochs):
    loss = lossfn(X, VISC)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100==0:
        LOSS.append(loss.item())
        print(f"Epoch: {i}\t\t Loss: {loss.item()}")

fig, ax = plt.subplots()
plt.plot(range(len(LOSS)), LOSS)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(f"./{NAME}/loss.png", bbox_inches = "tight")
plt.close('all')

VISC_ = model(X).detach().numpy()
VISC = VISC.detach().numpy()

fig, ax = plt.subplots()
plt.scatter(VISC_, VISC, ec='k')
plt.ylabel("Predicted values")
plt.xlabel("Measured values")
plt.axis('square')
plt.title(f'R2 = {r2_score(VISC, VISC_)  :.2f}')
plt.savefig(f"./{NAME}/plot.png", bbox_inches = "tight")
plt.close('all')

model.save(f"./{NAME}/visc.mmodel")