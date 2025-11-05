import drjit as dr
import drjit.nn as dnn
from drjit.opt import Adam

import drjit.auto as da
import numpy as np

data_folder = "bsdf-fixed-samples.npz"
data = np.load(data_folder)

print(data)