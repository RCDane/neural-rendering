import pickle
import numpy as np
def save_neural_network(weights , filename: str):
    as_np = np.array(weights)
    
    with open(filename, 'wb') as f:
        pickle.dump(as_np, f)

def load_neural_network(filename: str):
    with open(filename, 'rb') as f:
        as_np = pickle.load(f)
    return as_np