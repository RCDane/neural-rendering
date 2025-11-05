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




def main():
    args = arguments_parsing.parse_args()
    training_set, validation_set, _ = dataset_optimized.create_dataloaders_optimized(
        path=args.data,
        batch_size=10,
        num_workers=args.num_workers,
        pin_memory=True,
        use_batch_sampler=True,
    ) 

    inputsize = 10

    net = nn.Sequential(
        nn.Cast(Float16),
        nn.Linear(inputsize, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
        nn.Exp(),
        nn.Cast(Float32)
    )

    
    rng = dr.rng(seed=0)


    tranining_type = drad.TensorXf16

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
    
    

    for epoch in range(5):
        for b in training_set:
            wi_local = drad.TensorXf(b['wi_local'][0].T)
            wo_local = drad.TensorXf(b['wo_local'][0].T)
            # normal = dr.auto.TensorXf(b['shading_normal'])
            albedo = drad.TensorXf(b['albedo'][0].T)
            roughness = drad.TensorXf(b['roughness'][0].T)
            target = drad.Array3f(b['f'][0].T)

            input_tensor = drad.TensorXf16(dr.concat([
                wi_local[:],
                wo_local[:],
                albedo,
                roughness
            ], axis=0))
            # print(input_tensor.shape)

            input_tensor = dr.nn.CoopVec(input_tensor)
            pred = net(input_tensor)
            pred = drad.Array3f(pred)
            # print(f"pred: {pred}")
            loss = dr.mean(dr.square(pred - target))
            # loss = dr.auto.Float16(loss)
            # print("depends_on(weights)?", dr.depends_on(loss, weights))

            # print(f"loss: {loss}")
            
            dr.backward(scaler.scale(loss))
            scaler.step(opt)
            dr.set_grad(weights, 0)
            
    
if __name__ == "__main__":
    main()