import pickle
import numpy as np
import os
import json
import drjit as dr

"""
Supports layers of type:

- Linear
- ReLU
- Exp
- ScaleAdd
"""

def map_layer_to_meta(layer, weights=None):
    
    if type(layer) == dr.nn.Linear:
        return {
            'type': 'Linear',
            'config': layer.config,
            'shape': layer.weights.shape,
        }
    elif type(layer) == dr.nn.ReLU:
        return {
            'type': 'ReLU',
        }
    elif type(layer) == dr.nn.Exp:
        return {
            'type': 'Exp',
        }
    elif type(layer) == dr.nn.ScaleAdd:
        return {
            'type': 'ScaleAdd',
        }
        
def metadata_to_layer(meta):
    layer_type = meta['type']
    if layer_type == 'Linear':
        config = meta['config']
        layer = dr.nn.Linear(config[0], config[1], bias=config[3])
        return layer
    elif layer_type == 'ReLU':
        return dr.nn.ReLU()
    elif layer_type == 'Exp':
        return dr.nn.Exp()
    elif layer_type == 'ScaleAdd':
        return dr.nn.ScaleAdd(0.0, 0.0)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")




def save_neural_network(weights,network_architecture, filename: str):
    as_np = np.array(weights)
    print(weights)
    print(network_architecture)
    
    # metadata_list = []
    # for layer in network_architecture.layers:
    #     metadata_list.append(layer)
    
    # # if directory does not exist create it
    # path = os.path.dirname(filename)
    # if os.path.exists(path) == False:
    #     os.makedirs(path)
    
    
    
    # for metadata in metadata_list:
        
        
    
    layers = []
    # dump metadata as json
    for i,layer in enumerate(network_architecture.layers):
        layers.append({ i : map_layer_to_meta(layer)})
        
    print("Layers metadata:", layers) 
    
    metadata_path = filename + "_metadata.json"
    buffer_path = filename + "_weights.npy"
    metadata = {
        'layers': layers,
        'buffer_path:': buffer_path,        
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    with open(buffer_path, 'wb') as f:
        np.save(f, as_np)

def load_neural_network(metadata_path: str):
    
    metadata = None
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    buffer_path = metadata['buffer_path:']
    filename = os.path.dirname(metadata_path)
    filename = os.path.join(filename, buffer_path)
    
    
    with open(filename, 'rb') as f:
        as_np = np.load(f)
    
    
    layers = []
    for meta in metadata['layers']:
        for i, layer_meta in meta.items():
            layers.append(metadata_to_layer(layer_meta))
    
    print("Loaded layers:", layers)
    
    return 