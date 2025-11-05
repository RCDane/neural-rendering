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

import drjit.auto.ad as drad


# dr.set_backend("scalar")
# dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_flag(dr.JitFlag.Debug, True)          # Enables bounds checks & extra validation
dr.set_flag(dr.JitFlag.SymbolicScope, False)          # Optional: verbose kernel trace
dr.set_flag(dr.JitFlag.SymbolicCalls, False)
dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
dr.set_flag(dr.JitFlag.SymbolicScope, False)
def main():
    args = arguments_parsing.parse_args()
    samples = dataset_optimized.load_sample_file(args.data)

    # gather first 5000 samples for training
    import numpy as np
    # train_set = np.array([training_set.get for _ in range(1000)])
    
    print(f"Number of training samples: {len(samples['f'])}")

    inputsize = 10

    net = nn.Sequential(
        nn.Linear(inputsize, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
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
    
    opt = Adam(lr=1e-3, params={'weights': weights})

    scaler = GradScaler()

    res = 256
    
    batch_size = args.batch_size
    loss_avg = drad.Float32(0.0)
    counter = 0

    for epoch in tqdm(range(5), desc="Overall Training Progress", unit="epoch", total=5):
        loss_avg = drad.Float32(0.0)
        # for b in range(len(samples['f'])//batch_size):
        for b in tqdm(range(len(samples['f'])//batch_size), desc=f"Epoch {epoch+1}", total=len(samples['f'])//batch_size, unit="batch"):
            wi_local = samples['wi_local'][b*batch_size:(b+1)*batch_size]      # expect shape (3,)
            wo_local = samples['wo_local'][b*batch_size:(b+1)*batch_size]      # (3,)
            albedo   = samples['albedo'][b*batch_size:(b+1)*batch_size]        # (3,)
            r_val    = samples['roughness'][b*batch_size:(b+1)*batch_size]
            
            r_val = np.expand_dims(r_val, axis=-1)  # (1,)
            
            features = np.concatenate([wi_local, wo_local, albedo, r_val], axis=-1).T
            target = drad.TensorXf(samples['f'][b*batch_size:(b+1)*batch_size].T)
            
            input_tensor = drad.TensorXf16(features)

            input_tensor = dr.nn.CoopVec(input_tensor)
            pred = net(input_tensor)
            pred = drad.TensorXf(pred)
            loss = dr.mean(dr.square(pred - target))
            loss_avg += loss / (len(samples['f']) // batch_size)
            dr.backward(scaler.scale(loss))
            scaler.step(opt)
            dr.clear_grad(weights)
        
if __name__ == "__main__":
    main()