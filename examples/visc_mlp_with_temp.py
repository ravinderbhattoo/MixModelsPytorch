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
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./data/Visc_At_Tliq_SciGlass_1.csv")
NAME = "mlp_with_temp"
os.makedirs(f"{NAME}", exist_ok=True)

X = torch.FloatTensor(data.values[:, :-1])
X = torch.FloatTensor(StandardScaler().fit(X).transform(X))
VISC = torch.FloatTensor(data.values[:, -1:])

N = X.shape[1]

model = MModel() # separate components and temperature
NN = create_sequential_model([N, 30, 20, 1])  # will take composition as input

model.addmodel('nn', NN)

def func(X, F_):
    # use temperature
    x = F_.nn(X)
    return x

model.setforward(func)

def lossfn(X, y):
    visc_ = model(X)
    return ((visc_ - y)**2).mean()

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