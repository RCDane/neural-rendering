import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")  # or "llvm_ad_rgb" / "cuda_ad_rgb" for vectorization
    dr.set_backend("cuda")


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


def sample_face_uvs_multiple(face_id : dr.auto.ad.UInt, mesh : mi.Mesh, rng : dr.random.Generator, n: int):
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

def random_wi_sample(rng: dr.random.Generator, n: int =1):
    # Cosine-weighted hemisphere sampling
    sample_wi = mi.Point2f(rng.random(dr.auto.ad.Float, n), rng.random(dr.auto.ad.Float, n))
    wi_local = mi.warp.square_to_cosine_hemisphere(sample_wi)
    return wi_local

def generate_batched_uv_samples_for_importance_sampling(mesh: mi.Mesh, num_samples_per_face: int, metal_tex, rough_tex, base_tex, normal_tex, generator : dr.random.Generator = None):
    
    faces = mesh.faces_buffer()

    face_count = dr.width(faces) // 3

    indices = dr.arange(dr.auto.ad.UInt, face_count)
    areas = calculate_face_area(indices, mesh)
    area_total = dr.sum(areas)

    uv_samples = dr.zeros(dr.auto.ad.Array2f, shape= dr.width(indices))
    rng = generator
    # sample = sample_face_uvs(indices, mesh, rng)
    samples_multiple = sample_face_uvs_multiple(indices, mesh, rng, num_samples_per_face)
    # dr.eval(samples_multiple)





    bsdf = mesh.bsdf()

    si = mesh.eval_parameterization(mi.Point2f(samples_multiple), mi.RayFlags.UV)

    metalness = metal_tex.eval(si)
    roughness = rough_tex.eval(si)
    normal = normal_tex.eval(si)
    albedo = base_tex.eval(si)

    # wi_local = random_wi_sample(rng, dr.width(samples_multiple))
    # si.wi = wi_local

    # ctx = mi.BSDFContext()
    
    # s1 = generator.random(dr.auto.ad.Float, dr.width(samples_multiple))
    # s2 = mi.Point2f(generator.random(dr.auto.ad.Float, dr.width(samples_multiple)), generator.random(dr.auto.ad.Float, dr.width(samples_multiple)))
    
    # bs, spectrum = bsdf.sample(ctx, si, s1, s2)
    # print("spectrum", spectrum)
    
    # pdf = bs.pdf
    # dr.eval(samples_multiple, wi_local, ns, bsdf_val, metalness, roughness, albedo, normal, pdf)
    dr.eval(samples_multiple, metalness, roughness, albedo, normal, si)
    return samples_multiple, metalness, roughness, albedo,normal, si


def generate_batched_uv_samples(mesh: mi.Mesh, num_samples_per_face: int, metal_tex, rough_tex, base_tex, normal_tex, generator : dr.random.Generator = None):
    
    faces = mesh.faces_buffer()

    face_count = dr.width(faces) // 3

    indices = dr.arange(dr.auto.ad.UInt, face_count)
    areas = calculate_face_area(indices, mesh)
    area_total = dr.sum(areas)

    uv_samples = dr.zeros(dr.auto.ad.Array2f, shape= dr.width(indices))
    rng = generator
    # sample = sample_face_uvs(indices, mesh, rng)
    samples_multiple = sample_face_uvs_multiple(indices, mesh, rng, num_samples_per_face)
    # dr.eval(samples_multiple)





    bsdf = mesh.bsdf()

    si = mesh.eval_parameterization(mi.Point2f(samples_multiple), mi.RayFlags.UV)

    metalness = metal_tex.eval(si)
    roughness = rough_tex.eval(si)
    normal = normal_tex.eval(si)
    albedo = base_tex.eval(si)

    wi_local = random_wi_sample(rng, dr.width(samples_multiple))
    wo_local = random_wi_sample(rng, dr.width(samples_multiple))
    si.wi = wi_local

    ctx = mi.BSDFContext()
    
    bsdf_val = bsdf.eval(ctx, si, wo_local)
    
    pdf = bsdf.pdf(ctx, si, wi_local)
    dr.eval(samples_multiple, wi_local, wo_local, bsdf_val, metalness, roughness, albedo, normal, pdf)
    return samples_multiple, wi_local, wo_local, bsdf_val, metalness, roughness, albedo,normal, pdf, bsdf


def sample_specular(wi: mi.Vector3f,
                    alpha: mi.Vector3f,
                    slope: mi.Vector2f,
                    u: mi.Vector2f):
    rho = alpha.z
    sqrt_one_minus_rho = dr.sqrt(1.0 - rho * rho)

    denom = dr.sqrt(1.0 - u.x)
    s = dr.sqrt(u.x) / denom

    phi = (2.0 * dr.pi) * u.y
    sx_std = s * dr.cos(phi)
    sy_std = s * dr.sin(phi)

    sx = alpha.x * sx_std
    sy = alpha.y * (rho * sx_std + sqrt_one_minus_rho * sy_std)

    sx += slope.x
    sy += slope.y

    wh = dr.normalize(mi.Vector3f(-sx, -sy, 1.0))
    return 2.0 * dr.dot(wi, wh) * wh - wi


def sample_diffuse(slope: mi.Vector2f,
                   u: mi.Vector2f):
    wo_local = mi.warp.square_to_cosine_hemisphere(mi.Point2f(u))
    n = dr.normalize(mi.Vector3f(-slope.x, -slope.y, 1.0))
    frame = mi.Frame3f(n)
    return frame.to_world(wo_local)

def pdf_diffuse(slope: mi.Vector2f, wo: mi.Vector3f):
    n = dr.normalize(mi.Vector3f(-slope.x, -slope.y, 1.0))
    frame = mi.Frame3f(n)
    wo_local = frame.to_local(wo)
    return dr.maximum(0.0, wo_local.z * dr.inv_pi)

def pdf_specular(wi, wo, alpha, slope):
    eps = 1e-6
    wh = dr.normalize(wi + wo)
    sign = dr.select(wh.z >= 0, 1.0, -1.0)
    wh *= sign

    cos_theta = wh.z
    invalid = cos_theta <= 1e-4

    sx = -wh.x / dr.maximum(cos_theta, eps)
    sy = -wh.y / dr.maximum(cos_theta, eps)

    sx -= slope.x
    sy -= slope.y

    rho = alpha.z
    one_minus_rho2 = dr.maximum(0.0, 1.0 - rho * rho)
    sqrt_one_minus_rho = dr.sqrt(one_minus_rho2)

    normalization = dr.rcp(dr.maximum(alpha.x * alpha.y * sqrt_one_minus_rho, eps))

    sx_std = sx / alpha.x
    sy_std = (alpha.x * sy - rho * alpha.y * sx) * normalization

    r2 = sx_std * sx_std + sy_std * sy_std
    p22_std = dr.rcp(dr.pi * dr.square(1.0 + r2))
    p22 = p22_std * normalization

    pdf_h = p22 / dr.square(cos_theta) / cos_theta

    abs_dot = dr.abs(dr.dot(wi, wh))
    invalid |= abs_dot <= eps

    pdf = pdf_h / (4.0 * dr.maximum(abs_dot, eps))
    return dr.select(invalid, dr.zeros_like(pdf), pdf)
dr.syntax
def sample_analytic(
    alpha : dr.auto.ad.Array3f | dr.auto.ad.Array3f16, 
    slopeSpec : dr.auto.ad.Array2f | dr.auto.ad.Array2f16, 
    slopeDiff : dr.auto.ad.Array2f | dr.auto.ad.Array2f16,
    weightSpec : dr.auto.ad.Float | dr.auto.ad.Float16,
    wi : dr.auto.ad.Array3f | dr.auto.ad.Array3f16,
    u : dr.auto.ad.Array2f | dr.auto.ad.Array2f16,
    generator : dr.random.Generator = None):

    
    mask = u.x < weightSpec
    inv_weight_spec = dr.rcp(dr.maximum(weightSpec, 1e-8))
    inv_weight_diff = dr.rcp(dr.maximum(1.0 - weightSpec, 1e-8))

    u_spec = dr.auto.ad.Array2f(u)
    u_spec.x = u_spec.x * inv_weight_spec
    u_diff = dr.auto.ad.Array2f(u)
    u_diff.x = (u_diff.x - weightSpec) * inv_weight_diff

    wo_spec = sample_specular(wi, alpha, slopeSpec, u_spec)
    wo_diff = sample_diffuse(slopeDiff, u_diff)
    

    wo = dr.select(mask, wo_spec, wo_diff)

    pdf_spec = pdf_specular(wi, wo, alpha, slopeSpec)
    pdf_diff = pdf_diffuse(slopeDiff, wo)
    pdf = weightSpec * pdf_spec + (1.0 - weightSpec) * pdf_diff
    dr.eval(wo_spec, wo_diff, pdf_spec, pdf_diff, pdf)
    return wo_spec, wo_diff, pdf_spec, pdf_diff, pdf








def test():

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

    _tex_specs = {
        'normal': (f'{TEX}/lubricant_spray_nor_gl_1k.exr', True),
        'base':   (f'{TEX}/lubricant_spray_diff_1k.jpg', False),
        'metal':  (f'{TEX}/lubricant_spray_metal_1k.exr', True),
        'rough':  (f'{TEX}/lubricant_spray_rough_1k.exr', True),
    }

    def _load_tex(path: str, raw: bool) -> mi.Texture:
        return mi.load_dict({'type': 'bitmap', 'filename': path, 'raw': raw})

    _normal_tex, _base_tex, _metal_tex, _rough_tex = [ _load_tex(p, r) for p, r in _tex_specs.values() ]
    # ensure that samples are not the same between calls
    mesh : mi.Mesh = mi.load_dict(_scene_dict)
    
    generator = dr.rng(seed=dr.auto.ad.UInt(0))
    
    uv_coords, _, _, _, _, _, _ = generate_batched_uv_samples(
            mesh, 1, _metal_tex, _rough_tex, _base_tex, generator=generator)
    
    uv_coords2, _, _, _, _, _, _ = generate_batched_uv_samples(
            mesh, 1,  _metal_tex, _rough_tex, _base_tex, generator=generator)
    
    print("First 5 UV coords from first call:", uv_coords[:5])
    print("First 5 UV coords from second call:", uv_coords2[:5])
    assert not dr.allclose(uv_coords, uv_coords2), "UV coordinates from two calls should not be the same!"
    
if __name__ == "__main__":
    test()