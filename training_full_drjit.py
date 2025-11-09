import mitsuba as mi
import drjit as dr
dr.set_backend("cuda")
mi.set_variant('cuda_ad_rgb')
import drjit.nn as nn
from drjit.opt import Adam, GradScaler
from drjit.auto.ad import Texture2f, TensorXf, TensorXf16, Float16, Float32, Array2f, Array3f
from tqdm.auto import tqdm
from model import parse_model_input_values
import dataset_optimized
import arguments_parsing
import argparse
import numpy as np
import drjit.auto.ad as drad



# dr.set_flag(dr.JitFlag.Debug, True)          # Enables bounds checks & extra validation
# dr.set_flag(dr.JitFlag.SymbolicScope, False)          # Optional: verbose kernel trace
dr.set_flag(dr.JitFlag.SymbolicCalls, True)
dr.set_flag(dr.JitFlag.SymbolicConditionals, True)
# dr.set_flag(dr.JitFlag.SymbolicScope, True)
# dr.set_flag(dr.JitFlag.Debug, True)
def prepare_batch(b, samples, metal_tex, rough_tex, base_tex):
    wi_local = samples['wi_local'].T[b]      # expect shape (3,)
    wo_local = samples['wo_local'].T[b]      # (3,)
    uvs = samples['uv'].T[b]              # (2,)
    target = drad.TensorXf16(samples['bsdf_val'].T[b].T)
    
    wi_lo = dr.auto.ad.TensorXf16(wi_local)
    wo_lo = dr.auto.ad.TensorXf16(wo_local)

    si = mi.SurfaceInteraction3f()
    si.uv = mi.Point2f(uvs)
    metallic = drad.TensorXf16(metal_tex.eval(si).array)
    roughness = drad.TensorXf16(rough_tex.eval(si).array)
    albedo  = drad.TensorXf16(base_tex.eval(si).array)

    input_tensor = dr.nn.CoopVec(*wi_lo, *wo_lo, *metallic, *albedo)
    return input_tensor, target

def training_step(net, batch, target):

    pred = drad.TensorXf16(net(batch))
    loss = dr.mean(dr.square(pred - target))
    

    return loss

def run_epoch(net, opt, weights, scaler, samples, _metal_tex, _rough_tex, _base_tex):
    data_length = len(samples['uv'].T)
    avg_loss = 0.0
    for b in range(data_length):
        weights[:] = Float16(opt['weights'])

        input_tensor, target = prepare_batch(b, samples, _metal_tex, _rough_tex, _base_tex)
        loss = training_step(net, input_tensor, target)
        dr.backward(scaler.scale(loss))
        scaler.step(opt)
        avg_loss += loss
        dr.clear_grad(weights)
    print(f"Avg Loss: {avg_loss}")
        

def main():


    args = arguments_parsing.parse_args()
    samples = np.load(args.data, allow_pickle=True).item()



    OBJ = "data/lubricant_spray_1k.obj"
    TEX = "data/textures"

    _tex_specs = {
        'normal': (f'{TEX}/lubricant_spray_nor_gl_1k.exr', True),
        'base':   (f'{TEX}/lubricant_spray_diff_1k.jpg', False),
        'metal':  (f'{TEX}/lubricant_spray_metal_1k.exr', True),
        'rough':  (f'{TEX}/lubricant_spray_rough_1k.exr', True),
    }



    def _load_tex(path: str, raw: bool) -> mi.Texture:
        return mi.load_dict({'type': 'bitmap', 'filename': path, 'raw': raw})

    _normal_tex, _base_tex, _metal_tex, _rough_tex = [ _load_tex(p, r) for p, r in _tex_specs.values() ]

        # gather first 5000 samples for training
    # train_set = np.array([training_set.get for _ in range(1000)])
    data_length = len(samples['uv'].T)
    print(f"Number of training samples: {data_length}")

    inputsize = 12

    net = nn.Sequential(
        nn.Linear(inputsize, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Exp(),
    )


    rng = dr.rng(seed=drad.UInt(0))


    tranining_type = drad.TensorXf16
    print("Using training type:", tranining_type)
    net = net.alloc(
        dtype=tranining_type,
        size=-1,
        rng=rng
    )
    weights, net = nn.pack(net, layout='training')
    dr.enable_grad(weights)
    print(net)

    opt = Adam(lr=1e-3, params={'weights': Float32(weights)})
    scaler = GradScaler()

    res = 256

    loss_avg = drad.Float32(0.0)
    counter = 0


    epochs = 5

    for epoch in tqdm(range(epochs), desc="Overall Training Progress", unit="epoch", total=epochs):
        run_epoch(net, opt, weights, scaler, samples, _metal_tex, _rough_tex, _base_tex)    
if __name__ == "__main__":
    main()