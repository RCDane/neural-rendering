from tqdm.auto import tqdm
import imageio.v3 as iio
import drjit as dr
import drjit.nn as nn
from drjit.opt import Adam, GradScaler
from drjit.auto.ad import Texture2f, TensorXf, TensorXf16, Float16, Float32, Array2f, Array3f
import mitsuba as mi
dr.set_backend("cuda")
mi.set_variant('cuda_ad_rgb')
# Load a test image and construct a texture object
ref = TensorXf(iio.imread("https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/wave-128.png") / 256)
tex = Texture2f(ref)

# Establish the network structure
net = nn.Sequential(
    nn.TriEncode(16, 0.2),
    nn.Cast(Float16),
    nn.Linear(-1, -1, bias=False),
    nn.LeakyReLU(),
    nn.Linear(-1, -1, bias=False),
    nn.LeakyReLU(),
    nn.Linear(-1, -1, bias=False),
    nn.LeakyReLU(),
    nn.Linear(-1, 3, bias=False),
    nn.Exp()
)

# Instantiate a random number generator to initialize the network weights
rng = dr.rng(seed=0)

# Instantiate the network for a specific backend + input size
net = net.alloc(
    dtype=TensorXf16,
    size=2,
    rng=rng
)

# Convert to training-optimal layout
weights, net = nn.pack(net, layout='training')
print(net)

# Optimize a single-precision copy of the parameters
opt = Adam(lr=1e-3, params={'weights': Float32(weights)})

# This is an adaptive mixed-precision (AMP) optimization, where a half
# precision computation runs within a larger single-precision program.
# Gradient scaling is required to make this numerically well-behaved.
scaler = GradScaler()

res = 256
for i in range(10):
    for i in tqdm(range(500)):
        # Update network state from optimizer
        weights[:] = Float16(opt['weights'])

        # Generate jittered positions on [0, 1]^2
        t = dr.arange(Float32, res)
        p = (Array2f(dr.meshgrid(t, t)) + rng.random(Array2f, (2, res * res))) / res

        # Evaluate neural net + L2 loss
        
        coop = nn.CoopVec(p)
        pred = net(coop)
        img = Array3f(pred)
        loss = dr.squared_norm(tex.eval(p) - img)

        # Mixed-precision training: take suitably scaled steps
        dr.backward(scaler.scale(loss))
        scaler.step(opt)

# Done optimizing, now let's plot the result
t = dr.linspace(Float32, 0, 1, res)
p = Array2f(dr.meshgrid(t, t))
img = Array3f(net(nn.CoopVec(p)))

# Convert 'img' with shape 3 x (N*N) into a N x N x 3 tensor
img = dr.reshape(TensorXf(img, flip_axes=True), (res, res, 3))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(ref)
ax[1].imshow(dr.clip(img, 0, 1))
fig.tight_layout()
plt.show()