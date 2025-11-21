import os
import mitsuba as mi
import numpy as np
import drjit as dr
import matplotlib.pyplot as plt
# dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_flag(dr.JitFlag.SymbolicCalls, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
# dr.set_flag(dr.JitFlag.SymbolicLoops, False)
dr.set_thread_count(16)
mi.set_variant("cuda_ad_rgb")  # or "llvm_ad_rgb" / "cuda_ad_rgb" for vectorization
dr.set_backend("cuda")

import sampling
import drjit.auto.ad as drad
import drjit.nn as nn
from drjit.opt import Adam, GradScaler
from src.utils import save_neural_network, load_neural_network
import arguments_parsing



OBJ = "data/lubricant_spray_1k.obj"
TEX = "data/textures"
# Load mesh & create a scene or directly load a bsdf
_scene_dict = {
        'type': 'obj',
        'filename': OBJ,
        'face_normals': True,
        'bsdf': {
            'type': 'normalmap',
            'normalmap': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_nor_gl_1k.exr', 'raw': True},
            'bsdf': {
                'type': 'principled',
                'base_color': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_diff_1k.jpg', 'raw': False},
                'metallic':   {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_metal_1k.exr', 'raw': True},
                'roughness':  {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_rough_1k.exr', 'raw': True},
            }
        }
}

dr.syntax
def log_l1(pred, target, eps=1e-6):
    return dr.mean(dr.abs(pred - target))


import tqdm
import random
dr.syntax
def run_epoch(net, encoder_network, shading_frame_network, opt, weights, encoder_weights, shading_frame_weights, scaler,mesh, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator):
    # avg_loss = drad.Float32(0.0)
    # dr.disable_grad(avg_loss)
    # data_length = 1
    

    
    weights[:] = drad.Float16(opt['weights'])
    encoder_weights[:] = drad.Float16(opt['encoder_weights'])
    shading_frame_weights[:] = drad.Float16(opt['shading_frame_weights'])
    uv_coords, wi_local, wo_local, bsdf_val, metalness, roughness, albedo, normal = sampling.generate_batched_uv_samples(
        mesh, 1, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator=generator)
    
    
    dr.eval(uv_coords, wi_local, wo_local, bsdf_val, metalness, roughness, albedo, generator)
    
    metalness = drad.TensorXf16(metalness.x)
    roughness = drad.TensorXf16(roughness.x)
    albedo = drad.TensorXf16(albedo.array)

    encoded_texel = encoder_network(dr.nn.CoopVec(metalness, roughness, *albedo, *normal))
    
    unpacked_texel = drad.TensorXf16(encoded_texel)
    
    frame_vectors = shading_frame_network(encoded_texel)
    
    unpacked_frame_vectors = drad.TensorXf(frame_vectors) 
    
    sh_n = drad.Array3f(unpacked_frame_vectors[:3])
    sh_t = drad.Array3f(unpacked_frame_vectors[3:])
    sh_b = dr.cross(sh_n, sh_t)
    
    unpacked_shading_frame = drad.Matrix3f(sh_n, sh_t, sh_b)
    
    
    
    # print("Shading frame shape:", dr.shape(unpacked_shading_frame))
    
    dr.eval(unpacked_shading_frame)
    
    # print("wi shape:", dr.shape(wi))
    
    wi_transformed = dr.matmul(unpacked_shading_frame,drad.Array3f(wi_local))
    wo_transformed = dr.matmul(unpacked_shading_frame,drad.Array3f(wo_local))
    
    dr.eval(unpacked_texel)
    input_tensor = dr.nn.CoopVec(*drad.TensorXf16(wi_transformed), *drad.TensorXf16(wo_transformed), *unpacked_texel)
    
    
    
    target = drad.Array3f(bsdf_val)
    pred =  net(input_tensor)
    
    unpacked_pred = drad.TensorXf(pred)
    
    # print("Pred shape:", dr.shape(unpacked_pred))
    
    unpacked_color = drad.Array3f(unpacked_pred[:3])
    unpacked_albedo = drad.TensorXf(unpacked_pred[3:])
    # print("Target shape:", dr.shape(unpacked_color))
    # output is 3 bsdf and 3 rgb
    
        
    unpacked_color = dr.clip(unpacked_color, 0.0, 100000000.0)
    # dr.eval(sqr)
    # albedo_loss = dr.mean(dr.square(unpacked_albedo - albedo))
    bsdf_loss = log_l1(unpacked_color, target)
    
    loss = dr.mean(bsdf_loss)
    dr.backward(scaler.scale(loss))
    scaler.step(opt)
    avg_loss = loss
    # i += 1
    return avg_loss     

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network to approximate a BSDF using Mitsuba and Dr.Jit.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--output_folder', type=str, help='Folder for outputt')
    parser.add_argument('--training_config', type=str, help='Training configuration file path.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_arguments()
    
    TEX = "data/textures"

    _tex_specs = {
        'normal': (f'{TEX}/lubricant_spray_nor_gl_1k.exr', True),
        'base':   (f'{TEX}/lubricant_spray_diff_1k.jpg', False),
        'metal':  (f'{TEX}/lubricant_spray_metal_1k.exr', True),
        'rough':  (f'{TEX}/lubricant_spray_rough_1k.exr', True),
    }

    mesh : mi.Mesh = mi.load_dict(_scene_dict)

    params = mi.traverse(mesh)



    faces = mesh.faces_buffer()

    def _load_tex(path: str, raw: bool) -> mi.Texture:
        return mi.load_dict({'type': 'bitmap', 'filename': path, 'raw': raw})

    _normal_tex, _base_tex, _metal_tex, _rough_tex = [ _load_tex(p, r) for p, r in _tex_specs.values() ]

        # gather first 5000 samples for training
    # train_set = np.array([training_set.get for _ in range(1000)])

    encoder_input_size = 1 + 1 + 3 + 3  # metalness, roughness, albedo(3), normal(3)
    encoder_output_size = 8

    encoder_network = nn.Sequential(
        nn.Linear(encoder_input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, encoder_output_size),
    )
    
    

    inputsize = 11

    # 2 shading frames 3x3 matrices flattened = 18
    shading_frame_network = nn.Sequential(
        nn.Linear(encoder_output_size, 6),
    )
    
    net = nn.Sequential(
        nn.Linear(encoder_output_size + 6, 32),
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
    
    encoder_network = encoder_network.alloc(
        dtype=tranining_type,
        size=-1,
        rng=rng
    )
    
    shading_frame_network = shading_frame_network.alloc(
        dtype=tranining_type,
        size=-1,
        rng=rng
    )
    
    weights, net = nn.pack(net, layout='training')
    
    encoder_weights, encoder_network = nn.pack(encoder_network, layout='training')

    shading_frame_weights, shading_frame_network = nn.pack(shading_frame_network, layout='training')

    dr.enable_grad(weights)
    dr.enable_grad(encoder_weights)
    dr.enable_grad(shading_frame_weights)

    opt = Adam(
        lr=1e-4, 
        params={
            'weights': drad.Float32(weights), 
            'encoder_weights': drad.Float32(encoder_weights),
            'shading_frame_weights': drad.Float32(shading_frame_weights)
            }
        )
    scaler = GradScaler()

    res = 256

    loss_avg = drad.Float32(0.0)
    counter = 0

    save_every = 5
    save_folder = "trained_models/"
    run = "run1/"
    epochs = 5000

    generator = dr.rng(seed=drad.UInt(0))
    
    pbar = tqdm.tqdm( desc="Epoch Training Progress", unit="sample", total=epochs)
    print_every = 1000
    loss = drad.Float32(0.0)
    dr.disable_grad(mesh, _metal_tex, _rough_tex, _base_tex, _normal_tex)
    # dr.set_log_level(dr.LogLevel.Info)
    for epoch in range(epochs):
        i = dr.opaque(drad.Int,epoch)
        loss += run_epoch(net, encoder_network, shading_frame_network, opt, weights, encoder_weights, shading_frame_weights, scaler, mesh, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator=generator) 

        if i != 0 and i % print_every == 0:
            pbar.update(print_every)
            pbar.set_postfix({'loss': loss / print_every})
            loss = drad.Float32(0.0)

    from custom_bsdf_dr import NeuralBSDF
    
    neural_bsdf_dict = {
        "type": "neural_bsdf",
        'base_color': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_diff_1k.jpg', 'raw': False},
        'metallic':   {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_metal_1k.exr', 'raw': True},
        'roughness':  {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_rough_1k.exr', 'raw': True},
        'normal_map': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_nor_gl_1k.exr', 'raw': True},
        'input_dim': 11,
        'output_dim': 3,
    }
    neural_bsdf : NeuralBSDF = mi.load_dict(neural_bsdf_dict)
    
    dr.disable_grad(weights)
    dr.disable_grad(encoder_weights)
    dr.disable_grad(shading_frame_weights)
    neural_bsdf.add_model(net, encoder_network, shading_frame_network)
    scene_dict = {
		'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'env': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
        },
		'camera': {
			'type': 'perspective',
			'to_world': mi.ScalarTransform4f.look_at(
				origin=[0.2, 0.2, 0.2], target=[0, 0.1, 0], up=[0, 1, 0]
			),
			'fov': 45,
			'film': {
				'type': 'hdrfilm',
				'width': 400,
				'height': 400,
				'rfilter': {'type': 'box'}
			}
		},
		'mesh': 
        #     {

        #         'type': 'obj',
        #         'filename': OBJ,
        #         'face_normals': True,
        #         'bsdf': {
        #             'type': 'normalmap',
        #             'normalmap': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_nor_gl_1k.exr', 'raw': True},
        #             'bsdf': {
        #                 'type': 'principled',
        #                 'base_color': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_diff_1k.jpg', 'raw': False},
        #                 'metallic':   {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_metal_1k.exr', 'raw': True},
        #                 'roughness':  {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_rough_1k.exr', 'raw': True},
        #             }
        #         }
        # }
      {
			'type': 'obj',
            'filename': OBJ,
            'face_normals': True,
            'bsdf': neural_bsdf
		}
	}
    
    s = mi.load_dict(scene_dict)
    
    dr.set_flag(dr.JitFlag.SymbolicCalls, False)
    dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
    dr.set_flag(dr.JitFlag.SymbolicLoops, False)
    
    
    image = mi.render(s, spp=64, seed=np.random.randint(0, 10000)) / 11.0
    for i in range(10):
        image += mi.render(s, spp=64, seed=np.random.randint(0, 10000)) / 11.0
    
    
    print("Neural BSDF loaded.")
    print(neural_bsdf)

    # plt.axis("off")
    # plt.imshow(image)
    # plt.show()
    # verify model
    
    if args.output_folder is None:
        args.output_folder = "output"
    os.makedirs(args.output_folder, exist_ok=True)
    
    mi.util.write_bitmap(args.output_folder + '/render.png', image) 

if __name__ == "__main__":
    main()