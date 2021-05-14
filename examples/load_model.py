import dill
import torch
from MixModelsPytorch import MModel, load, save

model = load("mymodel.mmodel")

data = pd.read_csv("./code/data/Visc_At_Tliq_SciGlass_1.csv")

X = torch.FloatTensor(data.values[:, :-1])
y1 = torch.FloatTensor(data.values[:, -2:-1]) 
y2 = torch.FloatTensor(data.values[:, -1:])

print(model(X))
