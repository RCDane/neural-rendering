import os
import mitsuba as mi
import numpy as np
import drjit as dr
import matplotlib.pyplot as plt
# dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_flag(dr.JitFlag.SymbolicCalls, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
# dr.set_flag(dr.JitFlag.SymbolicLoops, False)
dr.set_thread_count(8)
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
        'face_normals': False,
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
def l1(pred, target, eps=1e-6):
    return dr.abs(pred - target)

dr.syntax
def log_l1(pred, target, eps=1e-6):
    target_safe = dr.maximum(target, eps)
    pred_safe = dr.maximum(pred, eps)
    
    return dr.mean(dr.abs(dr.log(pred_safe) - dr.log(target_safe)))

dr.syntax
def l2(pred, target):

    return dr.mean(dr.square(pred - target))

dr.syntax
def sigmoid(x):
    return 1 / (1 + dr.exp(-x))
    

dr.syntax
def kl_divergence(pred, target, eps=1e-9):
    """
    Computes the Kullback-Leibler divergence between two distributions.
    Assumes samples are drawn from the target distribution.
    D_KL(target || pred) = E_{x ~ target}[log(target(x)/pred(x))]
    """
    # Add a small epsilon to avoid log(0)
    target_safe = dr.maximum(target, eps)
    pred_safe = dr.maximum(pred, eps)
    
    # Only compute loss where the target PDF is non-zero
    loss = dr.abs(dr.log(target_safe) - dr.log(pred_safe))
    
    return dr.mean(loss)

import tqdm
import random

def run_shading_frame_network(network, input, wi, wo):
    if network is None:
        return wi, wo
    frame_vectors = network(input)
    
    unpacked_frame_vectors = drad.TensorXf(frame_vectors) 
    
    sh1_n = drad.Array3f(unpacked_frame_vectors[:3])
    sh1_t = drad.Array3f(unpacked_frame_vectors[3:6])
    sh2_n = drad.Array3f(unpacked_frame_vectors[6:9])
    sh2_t = drad.Array3f(unpacked_frame_vectors[9:12])
    
    
    sh1_b = dr.cross(sh1_n, sh1_t)
    sh2_b = dr.cross(sh2_n, sh2_t)
    unpacked_shading_frame1 = drad.Matrix3f(sh1_n, sh1_t, sh1_b)
    unpacked_shading_frame2 = drad.Matrix3f(sh2_n, sh2_t, sh2_b)
    
    
    wi_transformed = dr.matmul(unpacked_shading_frame1,drad.Array3f(wi))
    wo_transformed = dr.matmul(unpacked_shading_frame2,drad.Array3f(wo))
    dr.eval(wi_transformed, wo_transformed)
    return wi_transformed, wo_transformed


def train_decoder(
    net, 
    encoder_network, 
    shading_frame_network, 
    opt, 
    weights, 
    encoder_weights, 
    shading_frame_weights, 
    scaler, 
    mesh, 
    _metal_tex, 
    _rough_tex, 
    _base_tex, 
    _normal_tex, 
    generator,
):

    if net is not None:
        weights[:] = drad.Float16(opt['weights'])
    if encoder_network is not None:
        encoder_weights[:] = drad.Float16(opt['encoder_weights'])
    if shading_frame_network is not None:
        shading_frame_weights[:] = drad.Float16(opt['shading_frame_weights'])
    
    
    bsdf : mi.BSDF = mesh.bsdf()
    
    uv_coords, metalness, roughness, albedo, normal, si = sampling.generate_batched_uv_samples_for_importance_sampling(
    mesh, 1, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator)
    
    
    wi_local = sampling.random_wi_sample(generator, dr.width(uv_coords))
    si.wi = wi_local

    ctx = mi.BSDFContext()
    ctx.type_mask = mi.BSDFFlags.All
    
    s1 = generator.random(dr.auto.ad.Float, dr.width(uv_coords))
    s2 = mi.Point2f(generator.random(dr.auto.ad.Float, dr.width(uv_coords)), generator.random(dr.auto.ad.Float, dr.width(uv_coords)))
    
    bs, spectrum = bsdf.sample(ctx, si, s1, s2)
    
    pdf = bs.pdf
    wo_local = bs.wo

    
    
    metalness = drad.TensorXf16(metalness.x)
    roughness = drad.TensorXf16(roughness.x)
    albedo = drad.TensorXf16(albedo.array)

    encoded_texel = encoder_network(dr.nn.CoopVec(metalness, roughness, *albedo, *normal))
    
    unpacked_texel = drad.TensorXf16(encoded_texel)
    
    wi_transformed, wo_transformed = run_shading_frame_network(shading_frame_network, encoded_texel, wi_local, wo_local)
    


    
    dr.eval(unpacked_texel)
    input_tensor = dr.nn.CoopVec(*drad.TensorXf16(wi_transformed), *drad.TensorXf16(wo_transformed), *unpacked_texel)
    
    
    cos_theta_o = dr.maximum(mi.Frame3f.cos_theta(wo_local), 1e-6)

    pdf = dr.maximum(pdf, 1e-6)
    # print(spectrum)
    target = drad.Array3f(spectrum * pdf)
    pred =  net(input_tensor)
    
    unpacked_pred = drad.TensorXf(pred)

    unpacked_color = drad.Array3f(unpacked_pred[:3])
    
    

    bsdf_loss = log_l1(unpacked_color, target)
    loss = dr.mean(bsdf_loss) 

    dr.backward(scaler.scale(loss))
    scaler.step(opt)
    avg_loss = loss
    # i += 1
    return avg_loss




dr.syntax
def train_decoder_and_texture(
    net, 
    shading_frame_network, 
    opt, 
    weights, 
    shading_frame_weights, 
    scaler, 
    mesh, 
    latent_texture,
    generator : dr.random.Generator
):  
    weights[:] = drad.Float16(opt['weights'])
    shading_frame_weights[:] = drad.Float16(opt['shading_frame_weights'])
    
    rng = generator

    bsdf : mi.BSDF = mesh.bsdf()
    # n = latent_texture.shape[0]
    # uvs = generator.random(drad.Array2f, (2, n))

    # wi_local = mi.warp.square_to_cosine_hemisphere(uvs)

    # si.wi = wi_local
    ctx = mi.BSDFContext()
    # ctx.type_mask = mi.BSDFFlags.All
    faces = mesh.faces_buffer()

    face_count = dr.width(faces) // 3
    s1 = generator.random(dr.auto.ad.Float, dr.width(face_count))
    s2 = mi.Point2f(generator.random(dr.auto.ad.Float, dr.width(face_count)), generator.random(dr.auto.ad.Float, dr.width(face_count)))


    indices = dr.arange(dr.auto.ad.UInt, face_count)


    samples_multiple = sampling.sample_face_uvs_multiple(indices, mesh, rng, 1)    
    si = mesh.eval_parameterization(mi.Point2f(samples_multiple), mi.RayFlags.All)
    # print("si:", si)
    
    # sample = sample_face_uvs(indices, mesh, rng)

    
    bs, spectrum = bsdf.sample(ctx, si, s1, s2)
    
    wo_local = bs.wo

    
    
    latent_vals = latent_texture.eval(si.uv)
    
    
    frame_input = nn.CoopVec(drad.TensorXf16(latent_vals))
    
    
    wi_transformed, wo_transformed = run_shading_frame_network(shading_frame_network, frame_input, si.wi, wo_local)
    
    
    # print("latent_vector:", latent_vals)
    
    input_tensor = dr.nn.CoopVec(*drad.TensorXf16(wi_transformed), *drad.TensorXf16(wo_transformed), *drad.TensorXf16(latent_vals))
    
    pdf = bs.pdf
    # print("pdf:", pdf)
    # print("bs spectrum:", spectrum)
    cos_theta_o = dr.maximum(mi.Frame3f.cos_theta(wo_local), 1e-6)
    
    
    target = drad.Array3f(spectrum )
    

    pred =  net(input_tensor)

    unpacked_pred = drad.TensorXf(pred)

    unpacked_color = drad.Array3f(unpacked_pred[:3])
    # print("unpacked_color:", unpacked_color)


    bsdf_loss = log_l1(unpacked_color, target)
    # print("bsdf_loss:", dr.mean(bsdf_loss))
    loss = dr.mean(bsdf_loss) 
    # print("total loss:", loss)

    # i += 1
    return loss


dr.syntax
def train_sampler(
    importance_net,
    opt, 
    importance_weights, 
    scaler, 
    mesh,
    latent_texture,
    generator,
    epoch):

    
    
    importance_weights[:] = drad.Float16(opt['importance_weights'])
    
    
    bsdf : mi.BSDF = mesh.bsdf()
    
    n = latent_texture.shape[0]
    uvs = generator.random(drad.Array2f, (2, n))

    wi_local = mi.warp.square_to_cosine_hemisphere(uvs)
    si = mesh.eval_parameterization(mi.Point2f(uvs), mi.RayFlags.All)

    si.wi = wi_local
    si.uv = uvs
    ctx = mi.BSDFContext()
    ctx.type_mask = mi.BSDFFlags.All
    
    s1 = generator.random(dr.auto.ad.Float, dr.width(uvs))
    s2 = mi.Point2f(generator.random(dr.auto.ad.Float, dr.width(uvs)), generator.random(dr.auto.ad.Float, dr.width(uvs)))
    
    bs, _ = bsdf.sample(ctx, si, s1, s2)
    
    # # print("bs.pdf", bs.pdf)
    pdf = bs.pdf

    
    

    latent_vals = latent_texture.eval(si.uv)
    unpacked_texel = drad.TensorXf16(latent_vals)
    
    importance_input = dr.nn.CoopVec(*drad.TensorXf16(wi_local),*unpacked_texel)
    
    params = importance_net(importance_input)
    decoded = drad.TensorXf(params)

    alpha = mi.Vector3f( dr.exp(decoded[:3]-3.0))
    slope_spec = mi.Vector2f(dr.sinh(decoded[3:5]))
    slope_diff = mi.Vector2f(dr.sinh(decoded[5:7]))
    weight_spec = drad.Float(sigmoid(decoded[7]))
    
    s2 = mi.Point2f(generator.random(dr.auto.ad.Float, dr.width(uvs)), generator.random(dr.auto.ad.Float, dr.width(uvs)))

    wo_spec, wo_diff, pdf_spec, pdf_diff, pdf_pred = sampling.sample_analytic(
        drad.Array3f16(alpha), drad.Array2f16(slope_spec), drad.Array2f16(slope_diff), drad.Float16(weight_spec),
        drad.Array3f16(wi_local), drad.Array2f16(s2)
    )
    
    # Mix them using the predicted weight
    pdf_pred_at_gt = weight_spec * pdf_spec + (1.0 - weight_spec) * pdf_diff
    dr.eval(pdf_pred_at_gt)
    
    
    pdf_loss = log_l1(pdf, pdf_pred_at_gt)

    # i += 1
    return pdf_loss


# dr.syntax
# def run_epoch(
#     net, 
#     encoder_network, 
#     shading_frame_network, 
#     importance_net, 
#     opt, 
#     weights, 
#     encoder_weights, 
#     shading_frame_weights, 
#     importance_weights, 
#     scaler, 
#     mesh, 
#     _metal_tex, 
#     _rough_tex, 
#     _base_tex, 
#     _normal_tex, 
#     generator,
#     epoch):
#     # avg_loss = drad.Float32(0.0)
#     # dr.disable_grad(avg_loss)
#     # data_length = 1
    

    
#     weights[:] = drad.Float16(opt['weights'])
#     encoder_weights[:] = drad.Float16(opt['encoder_weights'])
#     shading_frame_weights[:] = drad.Float16(opt['shading_frame_weights'])
#     importance_weights[:] = drad.Float16(opt['importance_weights'])
    
    
#     bsdf : mi.BSDF = mesh.bsdf()
    
#     uv_coords, metalness, roughness, albedo, normal, si = sampling.generate_batched_uv_samples_for_importance_sampling(
#     mesh, 1, _metal_tex, _rough_tex, _base_tex, _normal_tex, generator)
    
    
#     wi_local = sampling.random_wi_sample(generator, dr.width(uv_coords))
#     si.wi = wi_local

#     ctx = mi.BSDFContext()
#     ctx.type_mask = mi.BSDFFlags.All
    
#     s1 = generator.random(dr.auto.ad.Float, dr.width(uv_coords))
#     s2 = mi.Point2f(generator.random(dr.auto.ad.Float, dr.width(uv_coords)), generator.random(dr.auto.ad.Float, dr.width(uv_coords)))
    
#     bs, spectrum = bsdf.sample(ctx, si, s1, s2)
    

#     is_gloss = dr.select(bs.sampled_type == 3, True, False)
#     # print("bs.pdf", bs.pdf)
#     pdf = bs.pdf
#     wo_local = bs.wo

    
    
#     metalness = drad.TensorXf16(metalness.x)
#     roughness = drad.TensorXf16(roughness.x)
#     albedo = drad.TensorXf16(albedo.array)

#     encoded_texel = encoder_network(dr.nn.CoopVec(metalness, roughness, *albedo, *normal))
    
#     unpacked_texel = drad.TensorXf16(encoded_texel)
    
#     importance_input = dr.nn.CoopVec(*unpacked_texel, wi_local)
    
#     params = importance_net(importance_input)
#     decoded = drad.TensorXf(params)

#     alpha = mi.Vector3f( dr.exp(decoded[:3]-3.0))
#     slope_spec = mi.Vector2f(dr.sinh(decoded[3:5]))
#     slope_diff = mi.Vector2f(dr.sinh(decoded[5:7]))
#     weight_spec = drad.Float(sigmoid(decoded[7]))
    

#     pdf_spec_eval = sampling.pdf_specular(si.wi, bs.wo, alpha, slope_spec)
#     pdf_diff_eval = sampling.pdf_diffuse(slope_diff, bs.wo)
    
#     # Mix them using the predicted weight
#     pdf_pred_at_gt = weight_spec * pdf_spec_eval + (1.0 - weight_spec) * pdf_diff_eval
#     dr.eval(pdf_pred_at_gt)
#     # wo_importance = dr.select(
#     #     is_gloss, 
#     #     sampling.sample_specular(si.wi, alpha, slope_spec, s2), 
#     #     sampling.sample_diffuse(slope_diff, s2))
#     # pdf_pred = dr.select(is_gloss, pdf_spec, pdf_diff)

    
#     frame_vectors = shading_frame_network(encoded_texel)
    
#     unpacked_frame_vectors = drad.TensorXf(frame_vectors) 
    
#     sh1_n = drad.Array3f(unpacked_frame_vectors[:3])
#     sh1_t = drad.Array3f(unpacked_frame_vectors[3:6])
#     sh2_n = drad.Array3f(unpacked_frame_vectors[6:9])
#     sh2_t = drad.Array3f(unpacked_frame_vectors[9:12])
    
    
#     sh1_b = dr.cross(sh1_n, sh1_t)
#     sh2_b = dr.cross(sh2_n, sh2_t)
#     unpacked_shading_frame1 = drad.Matrix3f(sh1_n, sh1_t, sh1_b)
#     unpacked_shading_frame2 = drad.Matrix3f(sh2_n, sh2_t, sh2_b)
    
    
#     dr.eval(unpacked_shading_frame1, unpacked_shading_frame2)
    

#     wi_transformed = dr.matmul(unpacked_shading_frame1,drad.Array3f(wi_local))
#     wo_transformed = dr.matmul(unpacked_shading_frame2,drad.Array3f(wo_local))
    
#     dr.eval(unpacked_texel)
#     input_tensor = dr.nn.CoopVec(*drad.TensorXf16(wi_transformed), *drad.TensorXf16(wo_transformed), *unpacked_texel)
    
    
    
#     # print(spectrum)
#     target = drad.Array3f(spectrum) 
#     pred =  net(input_tensor)
    
#     unpacked_pred = drad.TensorXf(pred)

#     unpacked_color = drad.Array3f(unpacked_pred[:3])
    
    
    
#     # pdf_loss = dr.mean(dr.square(pdf_pred - pdf))
#     # if epoch > 20000:
#     pdf_loss = kl_divergence(pdf, pdf_pred_at_gt)
#     # else:
#     #     pdf_loss = drad.Float(0.0)


#     bsdf_loss = log_l1(unpacked_color, target)
#     loss = dr.mean(bsdf_loss) 

#     # wo_loss = l2(wo_local, wo_importance)
    
#     # pdf_loss = dr.select(bs.pdf > 0.0, pdf_loss, drad.Float(0.0))
#     # wo_loss = dr.select(bs.pdf > 0.0, wo_loss, drad.Float(0.0))
#     # print("bsdf_loss:", bsdf_loss, "pdf_loss:", pdf_loss, "wo_loss:", wo_loss)
#     # wo_loss = 0.0
#     # loss = dr.mean(bsdf_loss)
#     dr.backward(scaler.scale(loss + pdf_loss))
#     scaler.step(opt)
#     avg_loss = loss
#     pdf_loss = dr.mean(pdf_loss)
#     # i += 1
#     return avg_loss, pdf_loss

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
    if 'decoder_epochs' in yaml_data:
        args.decoder_epochs = yaml_data['decoder_epochs']
    if 'importance_epochs' in yaml_data:
        args.importance_epochs = yaml_data['importance_epochs']
    if 'texture_epochs' in yaml_data:
        args.texture_epochs = yaml_data['texture_epochs']
    if 'texture_sample_size' in yaml_data:
        args.texture_sample_size = yaml_data['texture_sample_size']
    if 'output_folder' in yaml_data:
        args.output_folder = yaml_data['output_folder']
    if 'use_encoder' in yaml_data:
        args.use_encoder = yaml_data['use_encoder']
    if 'use_shading_frame' in yaml_data:
        args.use_shading_frame = yaml_data['use_shading_frame']
    if 'use_decoder' in yaml_data:
        args.use_decoder = yaml_data['use_decoder']
    if 'use_importance_sampling' in yaml_data:
        args.use_importance_sampling = yaml_data['use_importance_sampling']
    if 'use_texture' in yaml_data:
        args.use_texture = yaml_data['use_texture']
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
    parser.add_argument('--config', type=str, help='Training configuration file path.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--print_every', type=int, default=1000, help='Print training progress every N epochs.')
    parser.add_argument('--render_every', type=int, default=10000, help='Render test image every N epochs.')
    parser.add_argument('--decoder_epochs', type=int, default=20000, help='Number of epochs to train the decoder only.')
    parser.add_argument('--importance_epochs', type=int, default=50000, help='Number of epochs to train the importance network.')
    parser.add_argument('--texture_epochs', type=int, default=20000, help='Number of epochs to train the texture only.')
    parser.add_argument('--texture_sample_size', type=int, default=10000, help='Number of samples to use when training the texture.')
    parser.add_argument('--use_encoder', action='store_true', help='Whether to use the encoder network.')
    parser.add_argument('--use_shading_frame', action='store_true', help='Whether to use the shading frame network.')
    parser.add_argument('--use_decoder', action='store_true', help='Whether to use the decoder network.')
    parser.add_argument('--use_importance_sampling', action='store_true', help='Whether to use the importance sampling network.')
    parser.add_argument('--use_texture', action='store_true', help='Whether to use the latent texture.')
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
    
    
    config = parse_config_file(args.config, args)
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
        # params={
        #     'weights': drad.Float32(weights), 
        #     'encoder_weights': drad.Float32(encoder_weights),
        #     'shading_frame_weights': drad.Float32(shading_frame_weights),
        #     'importance_weights': drad.Float32(importance_weights)
        #     }
        )
    opt['weights'] = drad.Float32(weights)
    opt['encoder_weights'] = drad.Float32(encoder_weights)
    if args.use_shading_frame:
        opt['shading_frame_weights'] = drad.Float32(shading_frame_weights)

    
    # def cosine_learning_rate_scheduler(opt, min_lr, max_lr, epoch, max_epochs, warmup_epochs=1000):
    #     if epoch < warmup_epochs:
    #         lr = min_lr + (max_lr - min_lr) * (epoch + 1) / warmup_epochs
    #     else:
    #         progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    #         lr = 0.5 * (max_lr - min_lr) * (1 + dr.cos(dr.pi * progress)) + min_lr
    #     opt.set_learning_rate(lr)
    
    
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
    
    

    epochs = args.decoder_epochs

    generator = dr.rng(seed=drad.UInt(0))
    
    pbar = tqdm.tqdm( desc="Epoch Training Progress", unit="sample", total=epochs)
    print_every = args.print_every
    render_every = args.render_every
    loss = drad.Float32(0.0)
    dr.disable_grad(mesh, _metal_tex, _rough_tex, _base_tex, _normal_tex)
    # dr.set_log_level(dr.LogLevel.Info)
    
    loss_per_epoch = np.zeros(epochs//print_every)
    avg_loss = drad.Float32(0.0)
    
    if not args.use_shading_frame:
        shading_frame_network = None
        shading_frame_weights = None
    
    
    for epoch in range(epochs):
        loss = train_decoder(
            net, 
            encoder_network, 
            shading_frame_network, 
            opt, 
            weights, 
            encoder_weights, 
            shading_frame_weights, 
            scaler, 
            mesh, 
            _metal_tex, 
            _rough_tex, 
            _base_tex, 
            _normal_tex, 
            generator)
        avg_loss += loss
        if epoch != 0 and epoch % print_every == 0:
            pbar.update(print_every)
            pbar.set_postfix({'avg_loss': avg_loss / print_every, 'lr': opt.lr})
            # loss_per_epoch[epoch // print_every - 1] = avg_loss / print_every
            avg_loss = drad.Float32(0.0)
    
    def run_encoder(net,uv, metal_tex, rough_tex, base_tex, normal_tex):
        si = mi.SurfaceInteraction3f()
        si.uv = uv
        metalness = metal_tex.eval_1(si)
        roughness = rough_tex.eval_1(si)
        base_color = base_tex.eval(si)
        normal = normal_tex.eval(si)
        
        metalness = drad.TensorXf16(metalness)
        roughness = drad.TensorXf16(roughness)
        albedo = drad.TensorXf16(base_color.array)
        normal = drad.TensorXf16(normal.array)
        encoded_texel = encoder_network(dr.nn.CoopVec(metalness, roughness, *albedo, *normal))
        unpacked_texel = drad.TensorXf16(encoded_texel)
        return unpacked_texel
        
    
    def create_latent_texture(encoder_network, _metal_tex, _rough_tex, _base_tex, _normal_tex, resolution = (1024, 1024)):
        uv_texture = dr.meshgrid(dr.linspace(mi.Float, 0.0, 1.0, resolution[0]), dr.linspace(mi.Float, 0.0, 1.0, resolution[1]))
        
        latent_texture = run_encoder(encoder_network, uv_texture, _metal_tex, _rough_tex, _base_tex, _normal_tex)
        latent_texture = dr.reshape(latent_texture, (resolution[0], resolution[1], -1))
        return latent_texture
    res = 1024
    latent_texture : drad.TensorXf16 = create_latent_texture(encoder_network, _metal_tex, _rough_tex, _base_tex, _normal_tex, resolution=(res, res))
    # tex = mi.Bitmap(mi.Bitmap.PixelFormat.MultiChannel, mi.Struct.Type.Float16, (res, res))

    tex = latent_texture
    # dr.enable_grad(tex)
    
    # texture_sample_size = args.texture_sample_size
    # texture_epochs = args.texture_epochs
    # pbar = tqdm.tqdm( desc="Texture Training Progress", unit="sample", total=texture_epochs)
    # loss = drad.Float32(0.0)
    
    # dr.enable_grad(tex)
    
    # scaler1 = GradScaler()
    # scaler2 = GradScaler()
    # opt1 = Adam(
    #     lr=args.learning_rate,
    #     params={
    #         'weights': drad.Float32(weights),
    #         'shading_frame_weights': drad.Float32(shading_frame_weights),
    #         'texture': drad.TensorXf16(tex)
    #         }
    #     )
    # opt2 = Adam(
    #     lr=args.learning_rate,
    #     params={
    #         'importance_weights': drad.Float32(importance_weights)
    #         }
    #     )
    
    # generator = dr.rng(seed=drad.UInt(42))
    # # dr.disable_grad(weights)
    # avg_texture_loss = drad.Float32(0.0)
    # avg_importance_loss = drad.Float32(0.0)
    # for epoch in range(texture_epochs):
    #     i = dr.opaque(drad.Int,epoch)
    #     lt = drad.Texture2f16(opt1["texture"])
    #     texture_loss = train_decoder_and_texture(
    #         net,
    #         shading_frame_network,
    #         opt1,
    #         weights,
    #         shading_frame_weights,
    #         scaler,
    #         mesh,
    #         lt,
    #         generator)
    #     dr.backward(scaler1.scale(texture_loss))
    #     scaler1.step(opt1)
    #     # lt = drad.Texture2f16(opt1["texture"])
    #     # importance_loss = train_sampler(
    #     #     importance_net,
    #     #     opt2, 
    #     #     importance_weights, 
    #     #     scaler, 
    #     #     mesh, 
    #     #     lt,
    #     #     generator,
    #     #     epoch)
    #     # dr.backward(scaler2.scale(importance_loss))
    #     # scaler2.step(opt2)'
    #     avg_importance_loss += 0
    #     avg_texture_loss += texture_loss
    #     if i != 0 and i % print_every == 0:
    #         pbar.update(print_every)
    #         pbar.set_postfix({'texture loss': avg_texture_loss / print_every, 'importance loss': avg_importance_loss / print_every})

    #         avg_texture_loss = drad.Float32(0.0)
    #         avg_importance_loss = drad.Float32(0.0)


    # importance_epochs = args.importance_epochs
    # pbar = tqdm.tqdm( desc="Importance Sampling Progress", unit="sample", total=importance_epochs)
    # loss = drad.Float32(0.0)
    # lt = drad.Texture2f16(opt["texture"])

    
    
    
    # dr.disable_grad(weights)
    # avg_pdf_loss = drad.Float32(0.0)
    # for epoch in range(importance_epochs):
    #     i = dr.opaque(drad.Int,epoch)
    #     pdf_loss = train_sampler(
    #         importance_net,
    #         opt, 
    #         importance_weights, 
    #         scaler, 
    #         mesh, 
    #         lt,
    #         generator,
    #         epoch) 
    #     avg_pdf_loss += pdf_loss
    #     if i != 0 and i % print_every == 0:
    #         pbar.update(print_every)
    #         pbar.set_postfix({'Importance Sampling Loss': avg_pdf_loss / print_every})
    #         # loss_per_epoch[i // print_every - 1] = avg_loss / print_every
    #         avg_pdf_loss = drad.Float32(0.0)
        # if i != 0 and i % render_every == 0:
        #     neural_bsdf.add_model(net, encoder_network, shading_frame_network, importance_net)
        #     scene_dict = {
        #         'type': 'scene',
        #         'integrator': {
        #             'type': 'path'
        #         },
        #         'env': {
        #             'type': 'constant',
        #             'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
        #         },
        #         'camera': {
        #             'type': 'perspective',
        #             'to_world': mi.ScalarTransform4f.look_at(
        #                 origin=[0.2, 0.2, 0.2], target=[0, 0.1, 0], up=[0, 1, 0]
        #             ),
        #             'fov': 45,
        #             'film': {
        #                 'type': 'hdrfilm',
        #                 'width': 800,
        #                 'height': 800,
        #                 'rfilter': {'type': 'box'}
        #             }
        #         },
        #         'mesh': 
        #     {
        #             'type': 'obj',
        #             'filename': OBJ,
        #             'face_normals': True,
        #             'bsdf': neural_bsdf
        #         }
        #     }
            
        #     s = mi.load_dict(scene_dict)
        #     print("Rendering test image at epoch", i)
        #     image = render_image(s, expected_samples=64)
        #     if os.path.exists(args.output_folder) is False:
        #         args.output_folder = "output"
        #     if not os.path.exists(args.output_folder):
        #         os.makedirs(args.output_folder, exist_ok=True)
        #     mi.util.write_bitmap(args.output_folder + f'/render_epoch_{i}.png', image) 
            
            
            
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
    
    dec_net = net if args.use_decoder else None
    enc_net = encoder_network if args.use_encoder else None
    shading_net = shading_frame_network if args.use_shading_frame else None
    importance_net = importance_net if args.use_importance_sampling else None
    
    neural_bsdf.add_model(net, encoder_network, shading_frame_network, importance_net)
    
    texture = drad.Texture2f16(tex) if args.use_texture else None
    
    print("\nusing: decoder:", args.use_decoder, "encoder:", args.use_encoder, "shading frame:", args.use_shading_frame, "importance sampling:", args.use_importance_sampling, "texture:", args.use_texture, "\n")
    neural_bsdf.add_texture(texture)
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
            'face_normals': False,
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