import mitsuba as mi
import numpy as np
import drjit as dr

mi.set_variant("llvm_ad_rgb")  # or "llvm_ad_rgb" / "cuda_ad_rgb" for vectorization
dr.set_backend("llvm")
dr.set_flag(dr.JitFlag.Debug, True)

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
    print("shape:", faces)




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


mesh : mi.Mesh = mi.load_dict(_scene_dict)
print("Scene loaded.")
print("mesh:", mesh)
params = mi.traverse(mesh)
print("Parameters:", params)



faces = mesh.faces_buffer()
print(faces)
dr.syntax()
def surface_area(v0 : mi.Point3f, v1 : mi.Point3f, v2 : mi.Point3f):
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = dr.cross(edge1, edge2)
    area = 0.5 * dr.dot(cross_prod, cross_prod) ** 0.5
    return area




dr.syntax()
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

face_count = dr.width(faces) // 3
assert face_count == mesh.face_count()

indices = dr.arange(dr.auto.ad.UInt, face_count)
areas = calculate_face_area(indices, mesh)
area_total = dr.sum(areas)

uv_samples = dr.zeros(dr.auto.ad.Array2f, shape= dr.width(indices))
rng = dr.rng(seed = 42)
# sample = sample_face_uvs(indices, mesh, rng)
samples_multiple = sample_face_uvs_multiple(indices, mesh, rng, 1)
print(dr.sum(samples_multiple))
dr.eval(samples_multiple)

    # print("samples shape:", dr.shape(sample))
    # print("samples single:", sample)
    # print("sample multiple shape:", dr.shape(samples_multiple))
    # print("sample multiple:", samples_multiple)


# uv_samples = sample
# # uv_samples = sample_face_uvs(indices, mesh, mi.load_dict({"type": "independent", "sample_count":2}))
# # print("uv_samples: ",uv_samples)


def random_wi_sample(rng: dr.random.Generator, n: int =1):
    # Cosine-weighted hemisphere sampling
    sample_wi = mi.Point2f(rng.random(dr.auto.ad.Float, n), rng.random(dr.auto.ad.Float, n))
    wi_local = mi.warp.square_to_cosine_hemisphere(sample_wi)
    return wi_local


bsdf = mesh.bsdf()

si = mesh.eval_parameterization(mi.Point2f(samples_multiple), mi.RayFlags.All)


wi_local = random_wi_sample(rng, dr.width(samples_multiple))
wo_local = random_wi_sample(rng, dr.width(samples_multiple))
si.wi = wi_local

ctx = mi.BSDFContext()
bsdf_val = bsdf.eval(ctx, si, wo_local)
print("BSDF value shape:", bsdf_val)
print("is valid", dr.any(si.is_valid()))







def store_as_np(data, filename: str):
    np.save(filename, data)

store_as_np(data, "training_data.npy")

data_loaded = np.load("training_data.npy", allow_pickle=True).item()
print(data_loaded.keys())
print("Loaded UVs:", data_loaded["uv"])