import mitsuba as mi
import drjit as dr
import drjit as dr
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
# dr.set_flag(dr.JitFlag.SymbolicCalls, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
# dr.set_flag(dr.JitFlag.SymbolicScope, False)
def main():
    dr.set_backend("cuda")
    mi.set_variant('cuda_ad_rgb')
    dr.set_flag(dr.JitFlag.Debug, True)

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

    
    rng = dr.rng(seed=0)


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
    
    batch_size = args.batch_size
    loss_avg = drad.Float32(0.0)
    counter = 0

    
    epochs = 5

    for epoch in tqdm(range(epochs), desc="Overall Training Progress", unit="epoch", total=epochs):
        loss_avg = drad.Float32(0.0)
        print(f"Epoch {epoch+1} / {epochs}")
        # for b in range(len(samples['f'])//batch_size):
        for b in range(data_length):
            # print("0 \n")
            # weights[:] = drad.Float16(opt["weights"])
            # print("00 \n")
            wi_local = samples['wi_local'].T[b]      # expect shape (3,)
            wo_local = samples['wo_local'].T[b]      # (3,)
            uvs = samples['uv'].T[b]              # (2,)
            target = drad.TensorXf16(samples['bsdf_val'].T[b].T)

            wi_lo = dr.auto.ad.TensorXf(wi_local)
            wo_lo = dr.auto.ad.TensorXf(wo_local)
            # print("1 \n")
            si = mi.SurfaceInteraction3f()
            si.uv = mi.Point2f(uvs)
            metallic = _metal_tex.eval(si).array
            roughness = _rough_tex.eval(si).array
            albedo  = dr.ravel(_base_tex.eval(si).array)
            feat = dr.auto.ad.TensorXf16([
                dr.ravel(wi_lo),dr.ravel(wo_lo), dr.ravel(albedo), dr.ravel(roughness)
            ])
            feat = dr.reshape(feat, (1,inputsize))
            # feat = dr.reshape(feat, (1,inputsize))
            # print("2 \n")
            
            
            # input_tensor = drad.TensorXf16(feat)
            
            print("input_tensor:", feat)
            # input_tensor = dr.nn.CoopVec(feat)
            # print("input_tensor:", input_tensor)

            pred = net(feat)
            
            print("pred computed\n")
            
            pred_output = drad.Array3f(pred)
            print("pred shape:", pred_output, "\n")
            print("target shape:", target.shape)
            # print("3 \n")

            # print("pred:", pred, "target:", target)
            print("before loss")
            loss = dr.square(pred_output - target)
            print("after loss")
            # print("input:", input_tensor)
            # print("pred:", pred)
            # print("target:", target)
            # print("loss:", loss)
            # dr.eval(loss)
            # print("4 \n")
            dr.eval(loss)
            print("loss: ", loss)
            loss_avg += loss
            # print("5 \n")
            
            dr.backward(scaler.scale(loss))
            # print("6 \n")
            scaler.step(opt)
            # dr.clear_grad(weights)
        dr.eval(loss_avg)
        loss_avg /= 500
        print(f"Epoch {epoch+1} Loss: {loss_avg}")
if __name__ == "__main__":
    main()