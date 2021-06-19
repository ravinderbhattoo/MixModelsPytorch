import torch
import torch.nn as nn
import dill

class MLP(nn.Module):
    def __init__(self, D_in, D_out, activation=nn.ReLU, layers=[1], dropout_rate=[0.2], batch_norm=True, dropout=True):
        super(MLP, self).__init__()

        if len(layers)!=len(dropout_rate):
            dropout_rate = dropout_rate*len(layers)

        if type(activation)!=type([]):
            activation = [activation]*len(layers)

        self.seq = nn.Sequential()

        for a, b, p, n, act in zip([D_in]+layers[:-1], layers, dropout_rate, range(1+len(layers)), activation):

            self.seq.add_module("Linear {}".format(n), nn.Linear(a, b))

            if dropout:
                self.seq.add_module("Dropout {}".format(n), nn.Dropout(p=p))

            self.seq.add_module("Activation {}".format(n), act())

            if batch_norm:
                self.seq.add_module("Batch Norm {}".format(n), nn.BatchNorm1d(b))

        self.seq.add_module("Linear", nn.Linear(layers[-1], D_out))

    def forward(self,x):
        return self.seq(x)

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
        self.extra = {}

    def addmodel(self, name, model):
        self.models[name] = model
        self.params += nn.ParameterList(getattr(self.models, name).parameters())
        self.F_ = Dict2Class(self.models)

    def parameters(self):
        return self.params

    def __repr__(self,):
        str_ = f"""MModel
Catergories: {self.catergories}
Models: \n{self.F_}
Schema: \n{self.forwardfn.__doc__}
Extras: \n{list(self.extra.keys())}
Params: \n{[(i.shape, i.device) for i in self.params]}
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
             