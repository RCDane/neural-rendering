import mitsuba as mi, numpy as np
import drjit as dr

# Choose a fast variant (change to 'cuda_ad_rgb' if you built CUDA)
mi.set_variant('llvm_ad_rgb')

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

shape = scene.shapes()[0]
bsdf  = shape.bsdf()
ctx   = mi.BSDFContext()

# --- Vectorized helpers ---

def hemi_dir_dr(n,rng=None):
    if rng is None:
        rng = dr.rng()
    u1 = dr.full(mi.Float, 0, n)
    u2 = dr.full(mi.Float, 0, n)
    u1 = rng.random(mi.Float, shape=n)
    u2 = rng.random(mi.Float, shape=n)
    z  = u1
    r  = dr.sqrt(dr.fma(-z, z, 1.0))
    phi = 2.0 * np.pi * u2
    return mi.Vector3f(r*dr.cos(phi), r*dr.sin(phi), z)

def sample_bilinear_np(tex, uv):
    # uv: (N,2) in [0,1]; returns (N,C)
    h, w = tex.shape[:2]
    u = (uv[:,0] % 1.0) * (w - 1)
    v = (1 - (uv[:,1] % 1.0)) * (h - 1)
    x0 = np.floor(u).astype(np.int32); y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1);    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = (u - x0)[...,None]; dy = (v - y0)[...,None]
    c00 = tex[y0, x0]; c10 = tex[y0, x1]; c01 = tex[y1, x0]; c11 = tex[y1, x1]
    return (c00 * (1 - dx) * (1 - dy) +
            c10 * dx * (1 - dy) +
            c01 * (1 - dx) * dy +
            c11 * dx * dy)

def _load_tex(path, want_channels):
    bmp = mi.Bitmap(path)
    arr = np.array(bmp, copy=True)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] != want_channels:
        if arr.shape[2] == 1 and want_channels == 3:
            arr = np.repeat(arr, 3, axis=2)
        else:
            arr = arr[..., :want_channels]
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)

normal_tex = _load_tex(f'{TEX}/lubricant_spray_nor_gl_1k.exr', 3)
base_tex   = _load_tex(f'{TEX}/lubricant_spray_diff_1k.jpg', 3)
metal_tex  = _load_tex(f'{TEX}/lubricant_spray_metal_1k.exr', 1)
rough_tex  = _load_tex(f'{TEX}/lubricant_spray_rough_1k.exr', 1)

def generate_vectorized(n=1_000_000, chunk=500, out="bsdf_rgb_samples_extended.npz"):
    pos_list = []
    normal_list = []
    uv_list = []
    wi_list = []
    wo_list = []
    f_list = []
    pdf_list = []
    tangent_list = []
    normal_map_list = []
    shading_normal_list = []
    albedo_list = []
    rough_list = []
    metal_list = []
    rng = dr.rng()
    produced = 0
    while produced < n:
        m = min(chunk, n - produced)
    
        # Sample mesh positions (area sampling)
        # Generate m random points for position sampling
        sample_2 = mi.Point2f(rng.random(mi.Float, shape=m), rng.random(mi.Float, shape=m))
        ps = shape.sample_position(0.0, sample_2)

        si = mi.SurfaceInteraction3f()
        si.p         = ps.p
        si.n         = ps.n
        si.sh_frame  = mi.Frame3f(ps.n)
        si.uv        = ps.uv
        si.time      = 0.0

        wi_local = hemi_dir_dr(m)
        si.wi = wi_local

        # Random numbers for bsdf.sample
        s1 = rng.random(mi.Float, shape=m)
        s2 = rng.random(mi.Float, shape=m)
        samp, weight = bsdf.sample(ctx, si, s1, s2)
        wo_local = samp.wo

        # Evaluate
        f_val   = bsdf.eval(ctx, si, wo_local)
        pdf_val = bsdf.pdf(ctx, si, wo_local)

        # Fetch data to host (keep minimal)
        p_host   = np.array(si.p)
        n_host   = np.array(si.n)
        uv_host  = np.array(si.uv)
        wi_host  = np.array(wi_local)
        wo_host  = np.array(wo_local)
        f_host   = np.array(f_val)
        pdf_host = np.array(pdf_val)

        # Textures (NumPy vectorized)
        nm = sample_bilinear_np(normal_tex, uv_host)  # (m,3)
        bc = sample_bilinear_np(base_tex,   uv_host)
        rt = sample_bilinear_np(rough_tex,  uv_host)[:,0]
        mt = sample_bilinear_np(metal_tex,  uv_host)[:,0]

        # Tangent from shading frame
        tangent_host   = np.array(si.sh_frame.s)        # (m,3)
        bitangent_host = np.array(si.sh_frame.t)        # (m,3)
        n_host         = np.array(si.n) 
        # Transform normal map (assumes nm in [0,1], convert to -1..1 then normalize)
        
        nm_t = nm
        s_normal = (
            tangent_host * nm_t[:,0:1] +
            bitangent_host * nm_t[:,1:2] +
            n_host * nm_t[:,2:3]
        )
        s_normal /= (np.linalg.norm(s_normal, axis=1, keepdims=True) + 1e-8)

        pos_list.append(p_host)
        normal_list.append(n_host)
        uv_list.append(uv_host)
        wi_list.append(wi_host)
        wo_list.append(wo_host)
        f_list.append(f_host)
        pdf_list.append(pdf_host)
        tangent_list.append(tangent_host)
        normal_map_list.append(nm)
        shading_normal_list.append(s_normal)
        albedo_list.append(bc)
        rough_list.append(rt)
        metal_list.append(mt)

        produced += m
        if produced % (chunk*2) == 0 or produced == n:
            print(f"{produced} / {n}")

        # Release large temporaries
        dr.schedule()
        dr.eval()

    def cat(lst): return np.concatenate(lst, axis=0)

    np.savez_compressed(out,
                        pos=cat(pos_list),
                        normal=cat(normal_list),
                        uv=cat(uv_list),
                        wi_local=cat(wi_list),
                        wo_local=cat(wo_list),
                        f=cat(f_list),
                        pdf=cat(pdf_list),
                        tangent=cat(tangent_list),
                        normal_map=cat(normal_map_list),
                        shading_normal=cat(shading_normal_list),
                        albedo=cat(albedo_list),
                        roughness=cat(rough_list),
                        metallic=cat(metal_list))
    print("Saved", out)

# Original generate (kept for reference – consider removing); also fix incorrect call
def generate(n=2000000):
    # Slow scalar version (kept if needed)
    ...

if __name__ == "__main__":
    # Fixed: previously called generate(width, height, samples) which mismatched signature
    total = 1024 * 1024 * 4  # example target
    generate_vectorized(n=total)