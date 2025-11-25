import os
import mitsuba as mi
import numpy as np
import drjit as dr
import matplotlib.pyplot as plt
# dr.set_flag(dr.JitFlag.Debug, True)
dr.set_flag(dr.JitFlag.SymbolicCalls, False)
dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
dr.set_flag(dr.JitFlag.SymbolicLoops, False)
dr.set_thread_count(16)
mi.set_variant("cuda_ad_rgb")  # or "llvm_ad_rgb" / "cuda_ad_rgb" for vectorization
dr.set_backend("cuda")

import sampling
import drjit.auto.ad as drad
import drjit.nn as nn
from drjit.opt import Adam, GradScaler
from src.utils import save_neural_network, load_neural_network
import arguments_parsing

from custom_bsdf_dr import NeuralBSDF


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

dr.syntax()
def l1(pred, target, eps=1e-6):
    return dr.mean(dr.abs(pred - target))

dr.syntax()
def log_l1(pred, target, eps=1e-6):
    return dr.mean(dr.abs(dr.log(pred + eps) - dr.log(target + eps)))

dr.syntax()
def sigmoid(x):
    return 1 / (1 + dr.exp(-x))
    

import tqdm
import random
dr.syntax()
def run_epoch(
    net, 
    encoder_network, 
    shading_frame_network, 
    importance_net, 
    opt, 
    weights, 
    encoder_weights, 
    shading_frame_weights, 
    importance_weights, 
    scaler, 
    mesh, 
    _metal_tex, 
    _rough_tex, 
    _base_tex, 
    _normal_tex, 
    generator):
    # avg_loss = drad.Float32(0.0)
    # dr.disable_grad(avg_loss)
    # data_length = 1
    

    
    weights[:] = drad.Float16(opt['weights'])
    encoder_weights[:] = drad.Float16(opt['encoder_weights'])
    shading_frame_weights[:] = drad.Float16(opt['shading_frame_weights'])
    importance_weights[:] = drad.Float16(opt['importance_weights'])
    
    
    uv_coords, wi_local, wo_local, bsdf_val, metalness, roughness, albedo, normal, pdf, bsdf = sampling.generate_batched_uv_samples(
        mesh, 1, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator=generator)
    
    
    count = dr.width(pdf)
    
    # print("bsdf_val:", bsdf_val)
    # print("pdf:", pdf)
    metalness = drad.TensorXf16(metalness.x)
    roughness = drad.TensorXf16(roughness.x)
    albedo = drad.TensorXf16(albedo.array)

    encoded_texel = encoder_network(dr.nn.CoopVec(metalness, roughness, *albedo, *normal))
    
    unpacked_texel = drad.TensorXf16(encoded_texel)
    
    importance_input = dr.nn.CoopVec(*unpacked_texel, wi_local)
    
    params = importance_net(importance_input)
    decoded = drad.TensorXf(params)

    alpha = mi.Vector3f(dr.exp(decoded[:3]))
    slope_spec = mi.Vector2f(decoded[3:5])
    slope_diff = mi.Vector2f(decoded[5:7])
    weight_spec = sigmoid(decoded[7])
    u_rand = mi.Point2f(mi.Point2f(generator.random(drad.Float,  count), generator.random(drad.Float,  count)))

    wo_importance, pdf_pred = sampling.sample_analytic(
        alpha, slope_spec, slope_diff, weight_spec,
        drad.Array3f(wi_local), u_rand
    )
    ctx = mi.BSDFContext()
    pdf_pred = bsdf.pdf(ctx, mesh.eval_parameterization(mi.Point2f(uv_coords), mi.RayFlags.UV), wo_importance)
    
    frame_vectors = shading_frame_network(encoded_texel)
    
    unpacked_frame_vectors = drad.TensorXf(frame_vectors) 
    
    sh1_n = drad.Array3f(unpacked_frame_vectors[:3])
    sh1_t = drad.Array3f(unpacked_frame_vectors[3:6])
    sh2_n = drad.Array3f(unpacked_frame_vectors[6:9])
    sh2_t = drad.Array3f(unpacked_frame_vectors[9:12])
    
    
    sh1_b = dr.cross(sh1_n, sh1_t)
    sh2_b = dr.cross(sh2_n, sh2_t)
    unpacked_shading_frame1 = drad.Matrix3f(sh1_n, sh1_t, sh1_b)
    unpacked_shading_frame2 = drad.Matrix3f(sh2_n, sh2_t, sh2_b)
    
    
    
    # print("Shading frame shape:", dr.shape(unpacked_shading_frame))
    
    dr.eval(unpacked_shading_frame1, unpacked_shading_frame2)
    
    # print("wi shape:", dr.shape(wi))
    
    wi_transformed = dr.matmul(unpacked_shading_frame1,drad.Array3f(wi_local))
    wo_transformed = dr.matmul(unpacked_shading_frame2,drad.Array3f(wo_local))
    
    dr.eval(unpacked_texel)
    input_tensor = dr.nn.CoopVec(*drad.TensorXf16(wi_transformed), *drad.TensorXf16(wo_transformed), *unpacked_texel)
    
    
    
    
    target = drad.Array3f(bsdf_val) / (pdf + 1e-6)
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
    pdf_loss = dr.mean(dr.square(pdf_pred - pdf))
    loss = dr.mean(bsdf_loss) + dr.mean(pdf_loss) * 0.1
    # loss = dr.mean(bsdf_loss)
    dr.backward(scaler.scale(loss))
    scaler.step(opt)
    avg_loss = loss
    # i += 1
    return avg_loss     

import argparse
import yaml
def parse_config_file(config_path: str, args) -> dict:
    
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    if 'epochs' in yaml_data:
        args.epochs = yaml_data['epochs']
    if 'learning_rate' in yaml_data:
        args.learning_rate = yaml_data['learning_rate']
    if 'print_every' in yaml_data:
        args.print_every = yaml_data['print_every']
    if 'render_every' in yaml_data:
        args.render_every = yaml_data['render_every']
    return yaml_data
_LAYER_MAP = {
    'Linear': lambda args: nn.Linear(*args),
    'ReLU': lambda args: nn.ReLU(),
    'Exp': lambda args: nn.Exp(),
}

def build_network(layer_specs):
    layers = []
    for spec in layer_specs:
        (name, args), = spec.items()
        try:
            layers.append(_LAYER_MAP[name](args))
        except KeyError:
            raise ValueError(f"Unsupported layer '{name}' in config.")
    return nn.Sequential(*layers)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network to approximate a BSDF using Mitsuba and Dr.Jit.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--output_folder', type=str, help='Folder for outputt')
    parser.add_argument('--training_config', type=str, help='Training configuration file path.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--print_every', type=int, default=1000, help='Print training progress every N epochs.')
    parser.add_argument('--render_every', type=int, default=10000, help='Render test image every N epochs.')
    args = parser.parse_args()
    
    return args






def render_image(scene : mi.Scene, expected_samples: int):
    dr.set_flag(dr.JitFlag.SymbolicCalls, False)
    dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
    dr.set_flag(dr.JitFlag.SymbolicLoops, False)
    
    
    
    
    samples_per_pass = 16
    
    num_passes = float(expected_samples // samples_per_pass)
    
    render_bar = tqdm.tqdm( desc="Rendering Progress", unit="pass", total=int(num_passes))
    
    
    
    image = mi.render(scene, spp=samples_per_pass, seed=np.random.randint(0, 10000)) / num_passes
    for i in range(int(num_passes) - 1):
        image += mi.render(scene, spp=samples_per_pass, seed=np.random.randint(0, 10000)) / num_passes
        render_bar.update(1)
    render_bar.close()
    
    dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    dr.set_flag(dr.JitFlag.SymbolicConditionals, True)
    dr.set_flag(dr.JitFlag.SymbolicLoops, True)
    return image

def main():
    
    args = parse_arguments()
    
    
    config = parse_config_file(args.training_config, args)
    net_cfg = config['networks']
    

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


    
    
    importance_net = nn.Sequential(
        nn.Linear(encoder_output_size + 3, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 9),  # alpha_x, alpha_y, rho, slopeSpec(2x), slopeDiff(2x), weightSpec, weightDiff
    )
    

    rng = dr.rng(seed=drad.UInt(0))


    tranining_type = drad.TensorXf16
    print("Using training type:", tranining_type)
    
    encoder_network = build_network(net_cfg["encoder"]["layers"]).alloc(dtype=tranining_type, size=-1, rng=rng)
    shading_frame_network = build_network(net_cfg["shading_frame"]["layers"]).alloc(dtype=tranining_type, size=-1, rng=rng)
    net = build_network(net_cfg["bsdf"]["layers"]).alloc(dtype=tranining_type, size=-1, rng=rng)

    print("Encoder Network:", encoder_network)
    print("Shading Frame Network:", shading_frame_network)
    print("BSDF Network:", net)

    
    
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
    
    importance_net = importance_net.alloc(
        dtype=tranining_type,
        size=-1,
        rng=rng
    )
    
    
    
    weights, net = nn.pack(net, layout='training')
    
    encoder_weights, encoder_network = nn.pack(encoder_network, layout='training')

    shading_frame_weights, shading_frame_network = nn.pack(shading_frame_network, layout='training')
    importance_weights, importance_net = nn.pack(importance_net, layout='training')
    dr.enable_grad(weights)
    dr.enable_grad(encoder_weights)
    dr.enable_grad(shading_frame_weights)
    dr.enable_grad(importance_weights)
    opt = Adam(
        lr=args.learning_rate, 
        params={
            'weights': drad.Float32(weights), 
            'encoder_weights': drad.Float32(encoder_weights),
            'shading_frame_weights': drad.Float32(shading_frame_weights),
            'importance_weights': drad.Float32(importance_weights)
            }
        )
    scaler = GradScaler()

    res = 256

    loss_avg = drad.Float32(0.0)
    counter = 0

    save_every = 5
    save_folder = "trained_models/"
    run = "run1/"
    
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
    
    

    epochs = args.epochs

    generator = dr.rng(seed=drad.UInt(0))
    
    pbar = tqdm.tqdm( desc="Epoch Training Progress", unit="sample", total=epochs)
    print_every = args.print_every
    render_every = args.render_every
    loss = drad.Float32(0.0)
    dr.disable_grad(mesh, _metal_tex, _rough_tex, _base_tex, _normal_tex)
    # dr.set_log_level(dr.LogLevel.Info)
    
    loss_per_epoch = np.zeros(epochs//print_every)
    
    for epoch in range(epochs):
        i = dr.opaque(drad.Int,epoch)
        loss += run_epoch(net, encoder_network, shading_frame_network, importance_net, opt, weights, encoder_weights, shading_frame_weights, importance_weights, scaler, mesh, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator=generator) 

        if i != 0 and i % print_every == 0:
            pbar.update(print_every)
            pbar.set_postfix({'loss': loss / print_every})
            loss_per_epoch[i // print_every - 1] = loss / print_every
            loss = drad.Float32(0.0)
        if i != 0 and i % render_every == 0:
            neural_bsdf.add_model(net, encoder_network, shading_frame_network, importance_net)
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
                        'width': 800,
                        'height': 800,
                        'rfilter': {'type': 'box'}
                    }
                },
                'mesh': 
            {
                    'type': 'obj',
                    'filename': OBJ,
                    'face_normals': True,
                    'bsdf': neural_bsdf
                }
            }
            
            s = mi.load_dict(scene_dict)
            print("Rendering test image at epoch", i)
            image = render_image(s, expected_samples=64)
            if os.path.exists(args.output_folder) is False:
                args.output_folder = "output"
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder, exist_ok=True)
            mi.util.write_bitmap(args.output_folder + f'/render_epoch_{i}.png', image) 
            
            
            
    def save_data():
        if args.output_folder is None:
            output_folder = save_folder + run
        else:
            output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        # save_neural_network(
        #     net, 
        #     encoder_network, 
        #     shading_frame_network, 
        #     output_folder + '/neural_bsdf.drjit'
        # )
        
        plt.plot(loss_per_epoch)
        plt.xlabel('Epochs (x1000)')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.savefig(output_folder + '/loss_plot.png')
        plt.close()
        
        np.save(output_folder + '/loss_per_epoch.npy', np.array(loss_per_epoch))
        print(f"Saved trained model to {output_folder}/neural_bsdf.drjit")
    
    pbar.close()
    
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
				'width': 800,
				'height': 800,
				'rfilter': {'type': 'box'}
			}
		},
		'mesh': 
      {
			'type': 'obj',
            'filename': OBJ,
            'face_normals': True,
            'bsdf': neural_bsdf
		}
	}
    
    s = mi.load_dict(scene_dict)
    

    
    
    image = render_image(s, expected_samples=128)
    
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