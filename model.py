from unittest import case
import json
import os


def save_checkpoint(model, optimizer, epoch, training_loss, validation_loss, input_shape, path):
    payload = {
        'epoch': epoch,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'input_shape': input_shape,
        'hidden_dim' : model.hidden_dim,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(payload, path)

    # Write sidecar JSON for quick inspection
    meta_path = path + '.json'
    meta = {k: payload[k] for k in ['epoch', 'training_loss', 'validation_loss','input_shape', 'hidden_dim']}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[checkpoint] Saved: {path}")


def load_checkpoint(model, optimizer, path):
    if not os.path.isfile(path):
        print(f"[checkpoint] File not found: {path}")
        return 0, 0.0, 0.0
    payload = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(payload['model_state'])
    optimizer.load_state_dict(payload['optimizer_state'])
    epoch = payload.get('epoch', 0)
    training_loss = payload.get('training_loss', 0.0)
    validation_loss = payload.get('validation_loss', 0.0)
    print(f"[checkpoint] Loaded: {path} (epoch {epoch})")
    return epoch, training_loss, validation_loss

import numpy as np

def parse_model_input_values(input_sizes: list[tuple[str, int]], input_values: dict[str, any], batch_size: int) -> torch.Tensor:
    """
    Build input tensor in declared order (name,size) using batch dict.
    Accepts feature tensors shaped (B,F) or (1,B,F) and normalizes to (B,F).
    """
    device = None
    for v in input_values.values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break

    def _norm_vec(t: torch.Tensor) -> torch.Tensor:
        # Convert (1,B,F) -> (B,F)
        if t is not None and t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t

    wi = _norm_vec(input_values.get('wi_local'))
    wo = _norm_vec(input_values.get('wo_local'))
    normal = _norm_vec(input_values.get('shading_normal'))

    if wi is None or wo is None or normal is None:
        raise KeyError("Required keys: wi_local, wo_local, shading_normal")

    cols = []
    for name, size in input_sizes:
        val = input_values.get(name)
        if isinstance(val, torch.Tensor):
            val = _norm_vec(val)
            # Ensure (B,size)
            if val.dim() == 1 and val.shape[0] == batch_size:
                val = val.unsqueeze(-1)
            elif val.dim() > 2:
                val = val.view(batch_size, -1)
        elif isinstance(val, (list, tuple, np.ndarray)):
            val = torch.as_tensor(val, dtype=torch.float32, device=device)
            if val.dim() == 1:
                # Broadcast scalar feature vector to batch
                if val.shape[0] == size:
                    val = val.unsqueeze(0).expand(batch_size, -1)
                else:
                    raise ValueError(f"List/array for '{name}' has length {val.shape[0]} != expected {size}")
        elif isinstance(val, (int, float)):
            val = torch.full((batch_size, size), float(val), dtype=torch.float32, device=device)
        else:
            raise TypeError(f"Unsupported type for feature '{name}'")
        cols.append(val)
    input_tensor = torch.cat(cols, dim=-1)
    return input_tensor
# ...existing code...


# test parsing
def test_parse_model_input_values():
    input_size = [('ndotl', 1), ('ndotv', 1), ('albedo', 3), ('roughness', 1)]
    batch_size = 2
    wi_local = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    wo_local = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    albedo = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    roughness = torch.tensor([[[0.5], [0.8]]])
    input_values = {
        'wi_local': wi_local,
        'wo_local': wo_local,
        'shading_normal': normal,
        'albedo': albedo,
        'roughness': roughness
    }
    input_tensor = parse_model_input_values(input_size, input_values, batch_size)
    print(input_tensor)
# test_parse_model_input_values()

class TextureEncoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_layer_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_layer_dim)
        self.fc2 = torch.nn.Linear(hidden_layer_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class SimpleNeuralBSDF(torch.nn.Module):
    def __init__(self, input_layer_dim, hidden_layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_layer_dim
        self.fc1 = torch.nn.Linear(input_layer_dim, hidden_layer_dim)
        self.fc2 = torch.nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = torch.nn.Linear(hidden_layer_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


import drjit as dr
import drjit.nn as dnn


class TorchToDRWrapper(dr.CustomOp):
    def __init__(self, model: torch.nn.Module, input_size: int, target_dtype):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.target_dtype = target_dtype
        self.weights, self.drjit_model = self.wrap(model)
    def wrap(self, model:torch.nn.Module): 
        traced = torch.fx.symbolic_trace(model)
        drjit_modules = []
        # Map from module names to actual modules
        module_dict = dict(model.named_modules())
        for node in traced.graph.nodes:
            if node.op == 'call_module':
                torch_module = module_dict[node.target]
                if isinstance(torch_module, torch.nn.Linear):
                    drjit_module = dnn.Linear(torch_module.in_features, torch_module.out_features, bias=torch_module.bias is not None)
                    drjit_module.weights = self.target_dtype(torch_module.weight.detach().cpu().numpy())
                    if torch_module.bias is not None:
                        drjit_module.bias = self.target_dtype(torch_module.bias.detach().cpu().numpy())
                    else:
                        drjit_module.bias = self.target_dtype(np.zeros((torch_module.out_features,), dtype=np.float32))
                    drjit_modules.append(drjit_module)
                elif isinstance(torch_module, torch.nn.ReLU):
                    drjit_modules.append(dnn.ReLU())
        self.drjit_modules = drjit_modules
        drjit_model = dnn.Sequential(*drjit_modules)

        return dnn.pack(drjit_model, layout='inference')
    def eval(self, input):
        coopVec_input = dnn.CoopVec(input)
        packed_input = self.drjit_model(coopVec_input)
        return packed_input
    
    def print_weights(self):
        for w in self.drjit_modules:
            if isinstance(w, dnn.Linear):
                print(w.__str__())
                print(w.weights)
                print(w.bias)
            elif isinstance(w, dnn.ReLU):
                print(w.__str__())
                print("ReLU Layer")
    
    def __str__(self):
        return f"TorchToDRWrapper(drjit_model={self.drjit_model})"

    
if __name__ == "__main__":
    model = SimpleNeuralBSDF(input_layer_dim=3, hidden_layer_dim=2, output_dim=2)
    for param in model.parameters():
        print(param)
    # print(model)
    wrapper = TorchToDRWrapper(model, input_size=1, target_dtype=dr.auto.ad.TensorXf16)
    wrapper.print_weights()
    input = torch.randn((1,3))
    output = model(input)
    print("Torch output:", output)
    dr_input = wrapper.target_dtype(input.detach().cpu().numpy())
    dr_output = dr.auto.ad.TensorXf16(wrapper.eval(dr_input))
    print("DR output:", dr_output)