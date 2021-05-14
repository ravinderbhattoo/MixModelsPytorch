import torch
import torch.nn as nn
import dill

def create_sequential_model(layers, bias=True):
            model_layers = []
            for i in range(len(layers)-1):
                in_dim = layers[i]
                out_dim = layers[i+1]
                model_layers += [nn.Linear(in_dim, out_dim, bias=bias), nn.ReLU()]
            return nn.Sequential(*model_layers[:-1])

def save(filename, model):
    with open(filename, "wb+") as f:
        f.write(dill.dumps(model))

def load(filename):
    with open(filename, "rb") as f:
        return dill.loads(f.read())
             
class Dict2Class(object):
    def __init__(self, my_dict):
        self.names = list(my_dict.keys())
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def __str__(self):
        return "\n".join([f"{i}: {getattr(self,i)}\n" for i in self.names])

def println(*args):
    print(*args, "\n")

class MModel(nn.Module):
    def __init__(self, sizes=(-1,)):
        super(MModel, self).__init__()
        self.catergories = len(sizes)
        self.sizes = sizes
        self.models = nn.ModuleDict()
        self.F_ = Dict2Class(self.models)
        self.params = nn.ParameterList()

    def addmodel(self, name, model):
        self.models[name] = model
        self.F_ = Dict2Class(self.models)
        if isinstance(model, type(torch.nn.Module())):
            self.params += nn.ParameterList(getattr(self.F_, name).parameters())

    def parameters(self):
        return self.params

    def __repr__(self,):
        str_ = f"""MModel
Catergories: {self.catergories}
Models: \n{self.F_}
                """     
        return str_

    def __str__(self,):
        return self.__repr__()   

    def forward(self, X):
        if self.sizes == (-1,):
            return self.forwardfn(X, self.F_)
        else:
            xs = self.get_xs(X)
            return self.forwardfn(*xs, self.F_)

    def setforward(self, func):
        self.forwardfn = func

    def get_xs(self, X):
        start = 0
        out = []
        for i in self.sizes:
            out.append(X[:, start:start+i])
            start = start+i
        return out

    def save(self, filename):
        save(filename, self)
             