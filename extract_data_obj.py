import mitsuba as mi, numpy as np
import torch
mi.set_variant('llvm_ad_rgb')  # or 'llvm_rgb'

OBJ = "data/lubricant_spray_1k.obj"
TEX = "data/textures"

scene = mi.load_dict({
    'type': 'scene',
    'mesh': {
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
})
scene.shapes()[0]
import drjit as dr
dr.set_backend('llvm')
shape = scene.shapes()[0]
bsdf  = shape.bsdf()
ctx   = mi.BSDFContext()
rng = dr.rng()
def hemi_dir():
    """Generate one cosine-weighted(?) hemisphere direction (here uniform in z) using Dr.Jit."""
    u1 = rng.random(mi.Float, shape=1)
    u2 = rng.random(mi.Float, shape=1)
    z  = u1
    r  = dr.sqrt(dr.maximum(0.0, 1.0 - z * z))
    s, c = dr.sincos(2.0 * dr.pi * u2)
    return mi.Vector3f(r * c, r * s, z)

# Preload textures (robust to 1-channel or 3-channel sources)
def _load_tex(path, raw=True):
    # Instantiate the concrete 'bitmap' texture plugin
    return mi.load_dict({
        'type': 'bitmap',
        'filename': path,
        'raw': raw
    })


textures = {
    'normal': (f'{TEX}/lubricant_spray_nor_gl_1k.exr', True),
    'base':   (f'{TEX}/lubricant_spray_diff_1k.jpg', False),
    'metal':  (f'{TEX}/lubricant_spray_metal_1k.exr', True),
    'rough':  (f'{TEX}/lubricant_spray_rough_1k.exr', True),
}
textures = [ _load_tex(t, raw) for t, raw in textures.values() ]

normal_tex = textures[0]
base_tex   = textures[1]
metal_tex  = textures[2]
rough_tex  = textures[3]


def sample_bilinear(tex : mi.Texture, uv):
    return tex.eval(uv)

# We sample: wi, wo, uv, tangent, normal_map, albedo, roughness and metallic
def generate(n=2000000):
    pos = np.zeros((n, 3))
    normal = np.zeros((n, 3))
    uv = np.zeros((n, 2))
    wi = np.zeros((n, 3))
    wo = np.zeros((n, 3))
    f  = np.zeros((n, 3))
    pdf = np.zeros(n)
    tangent = np.zeros((n, 3))
    normal_map = np.zeros((n, 3))
    shading_normal = np.zeros((n, 3))
    albedo = np.zeros((n, 3))
    roughness_val = np.zeros(n)
    metallic_val = np.zeros(n)
    for i in range(n):
        samples = np.random.rand(2)
        ps = shape.sample_position(0.0, mi.Point2f(samples), True)
        si = mi.SurfaceInteraction3f()
        si.p = ps.p; si.n = ps.n; si.sh_frame = mi.Frame3f(ps.n); si.uv = ps.uv
        si.time = 0.0

        wi_local = hemi_dir()
        si.wi = wi_local
        samp, weight = bsdf.sample(ctx, si, mi.sample_tea_float(0, 1), mi.sample_tea_float(0, 1))
        wo_local = samp.wo

        f_val = bsdf.eval(ctx, si, wo_local)        # RGB BSDF value
        pdf_val = bsdf.pdf(ctx, si, wo_local)

        # Texture sampling
        nm = normal_tex.eval(si)
        # nm = nm * 2.0 - 1.0
        # nm /= (np.linalg.norm(nm) + 1e-8)
        normal_map[i] = np.array(nm).reshape(-1)

        pos[i]    = si.p.numpy().reshape(-1)
        normal[i] = si.n.numpy().reshape(-1)
        uv[i]     = si.uv.numpy().reshape(-1)
        wi[i]     = wi_local.numpy().reshape(-1)
        wo[i]     = wo_local.numpy().reshape(-1)
        f[i]      = f_val.numpy().reshape(-1)
        pdf[i]    = pdf_val.numpy().reshape(-1)

        # Tangent (use shading frame's first axis)
        tangent[i] = si.sh_frame.s.numpy().reshape(-1)

        bitangent = np.linalg.cross(normal[i],  tangent[i])
        n_tspace = normal_map[i]  # (nx, ny, nz) in tangent space
        s_normal = (
            tangent[i] * n_tspace[0:1] +
            bitangent * n_tspace[1:2] +
            normal[i] * n_tspace[2:3]
        )
        s_normal = s_normal / np.linalg.norm(s_normal)

        shading_normal[i] = s_normal


        bc = base_tex.eval(si).numpy().reshape(-1)
        albedo[i] = bc
        rough = rough_tex.eval(si)
        roughness_val[i] = rough[0].numpy()
        metallic_val[i]  = metal_tex.eval(si)[0].numpy()
        if (i+1) % 5000 == 0: print(i+1, "/", n)

    np.savez_compressed("bsdf_rgb_samples_extended.npz",
                        pos=pos, normal=normal, uv=uv,
                        wi_local=wi, wo_local=wo,
                        f=f, pdf=pdf,
                        tangent=tangent,
                        normal_map=normal_map,
                        shading_normal=shading_normal,
                        albedo=albedo,
                        roughness=roughness_val,
                        metallic=metallic_val)
    print("Saved bsdf_rgb_samples_extended.npz")

if __name__ == "__main__":
    width, height = 1024, 1024
    samples = 64
    print(width*height*samples)
    n = width * height * samples
    generate(n=n)
    
    
def test_tangent_orthogonality(infile="bsdf_rgb_samples_extended.npz"):
    data = np.load(infile)
    normal = data['normal']
    tangent = data['tangent']
    
    dot_products = np.sum(normal * tangent, axis=1)
    print("Max abs(dot(normal, tangent)):", np.max(np.abs(dot_products)))
    
    bitangent = np.cross(normal, tangent)
    det = np.linalg.det(np.stack([tangent, bitangent, normal], axis=-1))
    print("Min/Max determinant:", np.min(det), np.max(det))
    
# test_tangent_orthogonality()
# --- IGNORE ---