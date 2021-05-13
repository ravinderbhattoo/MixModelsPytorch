import dill
import torch
from MixModelsPytorch import MModel, load, save

model = load("mymodel.mmodel")

X = torch.FloatTensor([[1, 2, 3, 4], [1, 2, 3, 4]])
print(model(X))
