import mitsuba as mi
import numpy as np
import drjit as dr
# dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_flag(dr.JitFlag.SymbolicCalls, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
# dr.set_flag(dr.JitFlag.SymbolicLoops, False)
mi.set_variant("llvm_ad_rgb")  # or "llvm_ad_rgb" / "cuda_ad_rgb" for vectorization
dr.set_backend("llvm")
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

def sample_uvs_from_shape(s : mi.Mesh, sampler : mi.Sampler, num_samples: int):
    faces = s.faces_buffer()




dr.syntax()
def calculate_multiple_samples(s : mi.Shape, scene : mi.Scene, num_samples: int):
    sampler : mi.Sampler = mi.load_dict({"type": "independent", "sample_count": 10000})

    
    samples_2d = sampler.next_2d()          # shape (N, 2)
    pos_sample = s.sample_position(0.0, samples_2d)  # PositionSample3f
    si = s.eval_parameterization(pos_sample.uv, mi.RayFlags.All)
    bsdf = si.bsdf()

    hemisphere_samples = sampler.next_2d()   # New random samples
    wo_local = mi.warp.square_to_cosine_hemisphere(hemisphere_samples)  # shape (N, 3)

    wo_world = si.to_world(wo_local)


    ctx = mi.BSDFContext()


    bsdf_val = bsdf.eval(ctx, si, wo_local)


    bsdf_samples_2d = sampler.next_2d()
    bsdf_samples1_2d = sampler.next_1d()
    bsdf_val, spectrum = bsdf.sample(ctx, si, bsdf_samples1_2d ,bsdf_samples_2d)
    return bsdf_val, spectrum, si, wo_world


dr.syntax()
def calculate_sample(s : mi.Shape, scene : mi.Scene, sampler : mi.Sampler):
    samples_2d = sampler.next_2d()          # shape (N, 2)
    pos_sample = s.sample_position(0.0, samples_2d)  # PositionSample3f
    si = s.eval_parameterization(pos_sample.uv, mi.RayFlags.All)
    bsdf = si.bsdf()

    hemisphere_samples = sampler.next_2d()   # New random samples
    wo_local = mi.warp.square_to_cosine_hemisphere(hemisphere_samples)  # shape (N, 3)

    wo_world = si.to_world(wo_local)


    ctx = mi.BSDFContext()


    bsdf_val = bsdf.eval(ctx, si, wo_local)


    bsdf_samples_2d = sampler.next_2d()
    bsdf_samples1_2d = sampler.next_1d()
    bsdf_val, spectrum = bsdf.sample(ctx, si, bsdf_samples1_2d ,bsdf_samples_2d)
    return bsdf_val, spectrum, si, wo_world



def surface_area(v0 : mi.Point3f, v1 : mi.Point3f, v2 : mi.Point3f):
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = dr.cross(edge1, edge2)
    area = 0.5 * dr.dot(cross_prod, cross_prod) ** 0.5
    return area




def calculate_face_area(face_id : dr.auto.ad.UInt, mesh : mi.Mesh):
    face = mesh.face_indices(face_id)
    v0 : mi.Point3f = mesh.vertex_position(face[0])
    v1 : mi.Point3f = mesh.vertex_position(face[1])
    v2 : mi.Point3f = mesh.vertex_position(face[2])
    area = surface_area(v0, v1, v2)
    return area

def baricentric_sample(uv0 , uv1, uv2, sample1, sample2):

    sqrt_r1 = dr.sqrt(sample1)
    u = 1 - sqrt_r1
    v = sample2 * sqrt_r1

    uvs = (u * uv0 + v * uv1 + (1 - u - v) * uv2)
    return uvs


def sample_face_uvs(face_id : dr.auto.ad.UInt, mesh : mi.Mesh, rng : dr.random.Generator) -> mi.Vector2f | mi.Point2f:
    face = mesh.face_indices(face_id)
    uv0 = mesh.vertex_texcoord(face[0])
    uv1 = mesh.vertex_texcoord(face[1])
    uv2 = mesh.vertex_texcoord(face[2])
    # Sample UVs within the triangle defined by v0, v1, v2
    uvs = baricentric_sample(uv0, uv1, uv2, rng.random(dr.auto.ad.Float, 1), rng.random(dr.auto.ad.Float, 1))
    return uvs

def sample_face_uvs_multiple(face_id : dr.auto.ad.UInt, mesh : mi.Mesh, rng : dr.random.Generator, n: int) -> mi.Vector2f | mi.Point2f:
    face = mesh.face_indices(face_id)
    uv0 = mesh.vertex_texcoord(face[0])
    uv1 = mesh.vertex_texcoord(face[1])
    uv2 = mesh.vertex_texcoord(face[2])

    face_count = dr.width(face_id)

    # Sample positions within the triangle defined by v0, v1, v2
    uv0r = dr.reshape(mi.Point2f, dr.repeat(uv0, n), (2, n*face_count))
    uv1r = dr.reshape(mi.Point2f, dr.repeat(uv1, n), (2, n*face_count))
    uv2r = dr.reshape(mi.Point2f, dr.repeat(uv2, n), (2, n*face_count))

    s1 = rng.random(dr.auto.ad.Float, shape=n*dr.width(face_id))
    s2 = rng.random(dr.auto.ad.Float, shape=n*dr.width(face_id))
    
    positions = baricentric_sample(uv0r, uv1r, uv2r, s1, s2)
    
    return positions

# vertex_array_ordered = mi.Vector3f(dr.gather(dr.auto.ad.Float, vertices, faces))

def generate_batched_uv_samples(mesh: mi.Mesh, num_samples_per_face: int, metal_tex, rough_tex, base_tex):
    faces = mesh.faces_buffer()

    face_count = dr.width(faces) // 3
    assert face_count == mesh.face_count()

    indices = dr.arange(dr.auto.ad.UInt, face_count)
    areas = calculate_face_area(indices, mesh)
    area_total = dr.sum(areas)

    uv_samples = dr.zeros(dr.auto.ad.Array2f, shape= dr.width(indices))
    rng = dr.rng(seed = 42)
    # sample = sample_face_uvs(indices, mesh, rng)
    samples_multiple = sample_face_uvs_multiple(indices, mesh, rng, num_samples_per_face)
    dr.eval(samples_multiple)


    def random_wi_sample(rng: dr.random.Generator, n: int =1):
        # Cosine-weighted hemisphere sampling
        sample_wi = mi.Point2f(rng.random(dr.auto.ad.Float, n), rng.random(dr.auto.ad.Float, n))
        wi_local = mi.warp.square_to_cosine_hemisphere(sample_wi)
        return wi_local


    bsdf = mesh.bsdf()

    si = mesh.eval_parameterization(mi.Point2f(samples_multiple), mi.RayFlags.All)

    metalness = metal_tex.eval(si)
    roughness = rough_tex.eval(si)
    albedo = base_tex.eval(si)

    wi_local = random_wi_sample(rng, dr.width(samples_multiple))
    wo_local = random_wi_sample(rng, dr.width(samples_multiple))
    si.wi = wi_local

    ctx = mi.BSDFContext()
    bsdf_val = bsdf.eval(ctx, si, wo_local)
    dr.eval(samples_multiple, wi_local, wo_local, bsdf_val, metalness, roughness, albedo)
    # print("bsdf_val shape:", dr.shape(bsdf_val))
    # print("wi_local shape:", dr.shape(wi_local))
    return samples_multiple, wi_local, wo_local, bsdf_val, metalness, roughness, albedo





import tqdm

def run_epoch(net, opt, weights, scaler,mesh, _metal_tex, _rough_tex, _base_tex):
    avg_loss = drad.Float32(0.0)
    dr.disable_grad(avg_loss)
    data_length = 100
    for _ in range(data_length):
        weights[:] = drad.Float16(opt['weights'])
        # dr.enable_grad(weights)
        uv_coords, wi_local, wo_local, bsdf_val, metalness, roughness, albedo = generate_batched_uv_samples(mesh, 1, _metal_tex, _rough_tex, _base_tex)
        # print("wi_local type:", type(wi_local), " shape:", dr.shape(wi_local))
        # print("wo_local type:", type(wo_local), " shape:", dr.shape(wo_local))

        # wi_local = drad.TensorXf16(wi_local)
        # wo_local = drad.TensorXf16(wo_local)
        
        # dr.enable_grad(wi_local, wo_local, bsdf_val)
        metalness = drad.TensorXf16(metalness.array)
        roughness = drad.TensorXf16(roughness.array)
        albedo = drad.TensorXf16(albedo.array)
        # print("wi_local type:", type(wi_local), "wo_local type:", type(wo_local), "metalness type:", type(metalness), "roughness type:", type(roughness), "albedo type:", type(albedo))
        
        rng = dr.rng(seed=drad.UInt(0))
        

        
        input_tensor = dr.nn.CoopVec(*drad.TensorXf16(wi_local), *drad.TensorXf16(wo_local), *metalness, *roughness, *albedo)
        # input_tensor = dr.nn.CoopVec(drad.Array3f16(wi_local), drad.Array3f16(wo_local))
        
        target = drad.Array3f(bsdf_val)    
        pred =  net(input_tensor)
        unpacked_pred = drad.Array3f(pred)
        # print("pred shape:", dr.shape(unpacked_pred), " target shape:", dr.shape(target))
        sqr = dr.square(unpacked_pred - target)
        # print("sqr shape:", dr.shape(sqr))
        loss = dr.mean(sqr)
        # print("loss:", loss)
        dr.backward(scaler.scale(loss))
        scaler.step(opt)
        avg_loss += dr.mean(loss)
        # dr.clear_grad(weights)
    print(f"Avg Loss: {avg_loss/data_length}")
    
    
def main():
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


    inputsize = 15

    net = nn.Sequential(
        nn.Linear(inputsize, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.ScaleAdd(1.0, -3.0),
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
    
    
    save_neural_network(weights ,net, "initialized_model/trained_weights_initial.pkl")
    
    loaded_weights, loaded_net = load_neural_network("initialized_model/trained_weights_initial_metadata.json")
    

    dr.enable_grad(weights)

    opt = Adam(lr=1e-4, params={'weights': drad.Float32(weights)})
    scaler = GradScaler()

    res = 256

    loss_avg = drad.Float32(0.0)
    counter = 0

    save_every = 5
    save_folder = "trained_models/"
    run = "run1/"
    epochs = 100

    for epoch in tqdm.tqdm(range(epochs), desc="Overall Training Progress", unit="epoch", total=epochs):
        run_epoch(net, opt, weights, scaler, mesh, _metal_tex, _rough_tex, _base_tex) 
        # if epoch % save_every == 0:
        #     save_neural_network(weights ,net, f"{save_folder}{run}trained_weights_epoch_{epoch}.pkl")


    # verify model
    

if __name__ == "__main__":
    main()