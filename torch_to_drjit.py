import torch
import numpy as np
import drjit as dr
import drjit.nn as dnn
def convert_nn(nn:torch.nn.Module):
    param_dict = nn.state_dict()
    
    model = dr.nn.Sequential(
        dr.nn.Linear(10, 20),
        dr.nn.ReLU(),
        dr.nn.Linear(20, 5)
    )
    
    return model

def torch_sequential_to_drjit(torch_seq, input_size, tensor_type):
    # Build skeleton
    dj_layers = []
    for m in torch_seq:
        if isinstance(m, torch.nn.Linear):
            dj_layers.append(dnn.Linear(m.in_features, m.out_features, bias=m.bias is not None))
        elif isinstance(m, torch.nn.ReLU):
            dj_layers.append(dnn.ReLU())
        elif isinstance(m, torch.nn.Tanh):
            dj_layers.append(dnn.Tanh())
        else:
            raise NotImplementedError(f"Unsupported layer {type(m)}; extend this mapping.")
    dj_net = dnn.Sequential(*dj_layers).alloc(tensor_type, input_size)

    # Copy params
    ti = 0
    for m in torch_seq:
        if isinstance(m, torch.nn.Linear):
            dj_lin = dj_net.layers[ti]
            dj_lin.weights = tensor_type(m.weight.detach().cpu().numpy())
            if m.bias is not None:
                dj_lin.bias = tensor_type(m.bias.detach().cpu().numpy())
            ti += 1
        elif isinstance(m, (torch.nn.ReLU, torch.nn.Tanh)):
            ti += 1

    coeffs, dj_net = dnn.pack(dj_net, layout='evaluation')
    return coeffs, dj_net

# instantiate a simple neural network
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from drjit.interop import *

# create an instance of the neural network
nn = SimpleNN()
# convert the neural network to Dr.Jit format

wrapped = torch_sequential_to_drjit(nn, target_dtype=dr.auto.TensorXf)
# weights, drjit_nn = torch_sequential_to_drjit(nn.modules()., input_size=1, tensor_type=dr.auto.TensorXf16)
# drjit_nn = drjit_nn.alloc(dr.auto.TensorXf16, 1)
# weights, drjit_nn = dr.nn.pack(drjit_nn, layout='training')
rng = dr.rng(42)

raw_input = rng.random(dr.auto.TensorXf, (1, 10)).torch()
# input = dr.nn.CoopVec(raw_input)
output = wrapped(raw_input)
print(output.numpy())
nn_output = nn(torch.tensor(raw_input.cpu().numpy(), dtype=torch.float32))
print(nn_output)