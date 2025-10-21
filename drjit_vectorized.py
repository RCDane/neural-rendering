"""Vectorized BSDF sample generation using Dr.Jit/Mitsuba.

This splits the original per-sample loop in `extract_data_obj.generate` into
chunked vectorized evaluation to drastically reduce Python overhead.

Public functions:
  generate_data(n, chunk) -> dict[str, np.ndarray]
  save_npz(data, filename) -> None
  run_invariants(data) -> dict[str, float]
  test_vectorized(n, chunk) -> dict[str, float]

CLI usage:
  python drjit_vectorized.py --n 100000 --chunk 16384 --out bsdf_rgb_samples_ext_vec.npz
  python drjit_vectorized.py --test-only --n 5000
  python drjit_vectorized.py --n 2000000 --no-save

Assumptions:
  - Mitsuba variant 'llvm_ad_rgb' available. Switch to 'cuda_ad_rgb' if you
    have a CUDA build for further speed.
  - Memory sufficient for chosen chunk size (default 16384). Adjust if OOM.
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import mitsuba as mi
import drjit as dr

# -----------------------------------------------------------------------------
# Scene & plugin setup (mirrors extract_data_obj.py)
# -----------------------------------------------------------------------------
mi.set_variant('llvm_ad_rgb')  # change to 'cuda_ad_rgb' if GPU variant available

OBJ = "data/lubricant_spray_1k.obj"
TEX = "data/textures"

_scene_dict = {
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
}
scene = mi.load_dict(_scene_dict)
shape = scene.shapes()[0]
bsdf = shape.bsdf()
ctx = mi.BSDFContext()
_rng = dr.rng()  # single RNG instance reused

# Preload textures as concrete bitmap plugins
_tex_specs = {
    'normal': (f'{TEX}/lubricant_spray_nor_gl_1k.exr', True),
    'base':   (f'{TEX}/lubricant_spray_diff_1k.jpg', False),
    'metal':  (f'{TEX}/lubricant_spray_metal_1k.exr', True),
    'rough':  (f'{TEX}/lubricant_spray_rough_1k.exr', True),
}

def _load_tex(path: str, raw: bool):
    return mi.load_dict({'type': 'bitmap', 'filename': path, 'raw': raw})

_normal_tex, _base_tex, _metal_tex, _rough_tex = [ _load_tex(p, r) for p, r in _tex_specs.values() ]

# -----------------------------------------------------------------------------
# Core generation logic
# -----------------------------------------------------------------------------

def _sample_wi(m: int) -> mi.Vector3f:
    """Cosine-ish hemisphere distribution (currently uniform in cos(theta)=z)."""
    u1 = _rng.random(mi.Float, shape=m)
    u2 = _rng.random(mi.Float, shape=m)
    z = u1
    r = dr.sqrt(dr.maximum(0.0, 1.0 - z * z))
    s, c = dr.sincos(2.0 * dr.pi * u2)
    return mi.Vector3f(r * c, r * s, z)


def generate_data(n: int, chunk: int = 16384) -> dict:
    """Generate BSDF sample data in vectorized chunks.

    Parameters
    ----------
    n : int
        Total number of samples.
    chunk : int
        Number of samples processed per Dr.Jit batch.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary matching original `generate` keys.
    """
    # Accumulators (lists of arrays for later concatenation)
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
    ndotl_list = []
    ndotv_list = []
    ndoth_list = []
    ldoth_list = []

    produced = 0
    while produced < n:
        m = min(chunk, n - produced)

        # Sample surface positions (area sampling)
        u = _rng.random(mi.Float, shape=m)
        v = _rng.random(mi.Float, shape=m)
        sample_2 = mi.Point2f(u, v)
        ps = shape.sample_position(0.0, sample_2, True)

        si = mi.SurfaceInteraction3f()
        si.p = ps.p
        si.n = ps.n
        si.sh_frame = mi.Frame3f(ps.n)
        si.uv = ps.uv
        si.time = 0.0

        # Incoming direction sampling
        wi_local = _sample_wi(m)
        si.wi = wi_local

        # BSDF sampling (vectorized)
        s1 = _rng.random(mi.Float, shape=m)
        s2 = _rng.random(mi.Float, shape=m)
        samp, weight = bsdf.sample(ctx, si, s1, s2)
        wo_local = samp.wo

        # Evaluate BSDF & PDF (vectorized)
        f_val = bsdf.eval(ctx, si, wo_local)
        pdf_val = bsdf.pdf(ctx, si, wo_local)

        ndotl = dr.clip(wi_local.z, 0.0, 1.0)
        ndotv = dr.clip(wo_local.z, 0.0, 1.0)
        h = dr.normalize(wi_local + wo_local)
        ndoth = dr.clip(h.z, 0.0, 1.0)
        ldoth = dr.clip(dr.dot(wi_local, h), 0.0, 1.0)
        
        # Schedule & evaluate to make results ready for host transfer
        dr.schedule(f_val, pdf_val, si.p, si.n, si.uv, wi_local, wo_local, si.sh_frame.s, ndotl, ndotv, ndoth, ldoth)
        dr.eval()
        
        

        # Texture sampling (vectorized). Each returns a drjit array.
        nm = _normal_tex.eval(si)       # (m, 3)
        bc = _base_tex.eval(si)         # (m, 3)
        rt = _rough_tex.eval(si)        # (m, C) treat first channel
        mt = _metal_tex.eval(si)        # (m, C) treat first channel
        dr.schedule(nm, bc, rt, mt)
        dr.eval()

        # Host conversion (bulk)
        p_host = np.array(si.p)
        n_host = np.array(si.n)
        def _to_rows(a: np.ndarray, dim: int, expected_len: int) -> np.ndarray:
            """Return array with shape (expected_len, dim).

            Accepts input shapes:
              (dim, expected_len) -> transpose
              (expected_len, dim) -> pass-through
              (dim,) with expected_len==1 -> reshape
              (expected_len,) with dim==1 -> reshape (rare)
            Raises if shape incompatible.
            """
            a = np.array(a)
            if a.ndim == 2:
                if a.shape == (dim, expected_len):
                    return a.T
                if a.shape == (expected_len, dim):
                    return a
                # Some Dr.Jit arrays may come as (dim, expected_len, 1)
                if a.shape == (dim, expected_len, 1):
                    return a[:, :, 0].T
                if a.shape == (expected_len, dim, 1):
                    return a[:, :, 0]
            if a.ndim == 1:
                if a.shape[0] == dim and expected_len == 1:
                    return a.reshape(1, dim)
                if a.shape[0] == expected_len and dim == 1:
                    return a.reshape(expected_len, 1)
            raise ValueError(f"Cannot orient array of shape {a.shape} to (m={expected_len}, {dim})")

        p_host = _to_rows(np.array(si.p), 3, m)
        n_host = _to_rows(np.array(si.n), 3, m)
        uv_host = _to_rows(np.array(si.uv), 2, m)
        wi_host = _to_rows(np.array(wi_local), 3, m)
        wo_host = _to_rows(np.array(wo_local), 3, m)
        f_host = _to_rows(np.array(f_val), 3, m)
        pdf_host = np.array(pdf_val)  # already (m,)
        tangent_host = _to_rows(np.array(si.sh_frame.s), 3, m)
        bitangent_host = _to_rows(np.array(si.sh_frame.t), 3, m)  # second axis from frame
        def _scalar_to_col(a):
            a_np = np.array(a).reshape(-1)
            return a_np[:, None]

        ndotl_list.append(_scalar_to_col(ndotl))
        ndotv_list.append(_scalar_to_col(ndotv))
        ndoth_list.append(_scalar_to_col(ndoth))
        ldoth_list.append(_scalar_to_col(ldoth))
        nm_host = _to_rows(np.array(nm), 3, m)
        bc_host = _to_rows(np.array(bc), 3, m)
        # Roughness & metallic: take first channel; ensure shape (m,)
        rt_arr = np.array(rt)
        mt_arr = np.array(mt)
        def _scalar_channel(a: np.ndarray, m_expected: int) -> np.ndarray:
            """Return 1D array of length m_expected from various 2D layouts.

            Handles shapes:
              (m, 1) -> squeeze
              (1, m) -> squeeze
              (m, C) -> take first column
              (C, m) with small C (<=4) -> take first row (common bitmap layout channels x samples)
              (m,) already fine
            """
            if a.ndim == 2:
                if a.shape == (m_expected, 1):
                    return a[:, 0]
                if a.shape == (1, m_expected):
                    return a[0]
                # Channel-major: (C, m)
                if a.shape[1] == m_expected and a.shape[0] <= 4:
                    return a[0]
                # Sample-major: (m, C)
                if a.shape[0] == m_expected and a.shape[1] >= 1:
                    return a[:, 0]
            return a.reshape(-1)
        rt_host = _scalar_channel(rt_arr, m)
        mt_host = _scalar_channel(mt_arr, m)

        # Transform normal map (tangent space -> shading space)
        s_normal = (
            tangent_host * nm_host[:, 0:1] +
            bitangent_host * nm_host[:, 1:2] +
            n_host * nm_host[:, 2:3]
        )
        s_normal /= (np.linalg.norm(s_normal, axis=1, keepdims=True) + 1e-8)

        # Append chunk results
        pos_list.append(p_host)
        normal_list.append(n_host)
        uv_list.append(uv_host)
        wi_list.append(wi_host)
        wo_list.append(wo_host)
        f_list.append(f_host)
        pdf_list.append(pdf_host)
        tangent_list.append(tangent_host)
        normal_map_list.append(nm_host)
        shading_normal_list.append(s_normal)
        albedo_list.append(bc_host)
        rough_list.append(rt_host)
        metal_list.append(mt_host)

        produced += m
        if produced % (chunk * 5) == 0 or produced == n:
            print(f"Generated {produced} / {n}")

    # Concatenate
    def cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.empty((0,))

    data = {
        'pos': cat(pos_list),
        'normal': cat(normal_list),
        'uv': cat(uv_list),
        'wi_local': cat(wi_list),
        'wo_local': cat(wo_list),
        'f': cat(f_list),
        'pdf': cat(pdf_list),
        'tangent': cat(tangent_list),
        'normal_map': cat(normal_map_list),
        'shading_normal': cat(shading_normal_list),
        'albedo': cat(albedo_list),
        'roughness': cat(rough_list),
        'metallic': cat(metal_list),
        'ndotl': cat(ndotl_list),
        'ndotv': cat(ndotv_list),
        'ndoth': cat(ndoth_list),
        'ldoth': cat(ldoth_list),
    }
    return data

# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

def save_npz(data: dict, filename: str, compress: bool = True):
    if compress:
        np.savez_compressed(filename, **data)
    else:
        np.savez(filename, **data)
    kind = "compressed" if compress else "uncompressed"
    print(f"Saved {kind} file: {filename}")

# -----------------------------------------------------------------------------
# Tests & invariants
# -----------------------------------------------------------------------------

def run_invariants(data: dict) -> dict:
    """Compute diagnostic metrics to validate generated data integrity.

    Additional shape checks ensure 2D arrays have the expected per-sample dimensionality.
    """
    metrics = {}
    n = next(iter(data.values())).shape[0] if data else 0
    # Shape checks (length consistency)
    for k, arr in data.items():
        if arr.shape[0] != n:
            raise ValueError(f"Array {k} has inconsistent length {arr.shape[0]} vs {n}")

    # Expected second-dimension sizes for vector quantities
    expected_dims = {
        'pos': 3,
        'normal': 3,
        'uv': 2,
        'wi_local': 3,
        'wo_local': 3,
        'f': 3,
        'tangent': 3,
        'normal_map': 3,
        'shading_normal': 3,
        'albedo': 3,
    }
    for k, dim in expected_dims.items():
        arr = data[k]
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"Array {k} expected shape (n,{dim}) but got {arr.shape}")

    # Tangent orthogonality
    tan = data['tangent']
    nor = data['normal']
    dot_nt = np.sum(tan * nor, axis=1)
    metrics['tangent_normal_dot_max_abs'] = float(np.max(np.abs(dot_nt)))
    metrics['tangent_length_mean'] = float(np.mean(np.linalg.norm(tan, axis=1)))

    # Shading normal unit length
    s_n = data['shading_normal']
    s_len = np.linalg.norm(s_n, axis=1)
    metrics['shading_normal_len_mean'] = float(np.mean(s_len))
    metrics['shading_normal_len_max_dev'] = float(np.max(np.abs(s_len - 1.0)))

    # PDF non-negativity
    pdf = data['pdf']
    metrics['pdf_min'] = float(np.min(pdf))
    metrics['pdf_mean'] = float(np.mean(pdf))

    # UV bounds
    uv = data['uv']
    metrics['uv_min'] = float(np.min(uv))
    metrics['uv_max'] = float(np.max(uv))

    # BSDF value non-negativity
    f = data['f']
    metrics['f_min'] = float(np.min(f))
    metrics['f_max'] = float(np.max(f))

    # Roughness range
    rough = data['roughness']
    metrics['roughness_min'] = float(np.min(rough))
    metrics['roughness_max'] = float(np.max(rough))

    # Metallic range
    metal = data['metallic']
    metrics['metallic_min'] = float(np.min(metal))
    metrics['metallic_max'] = float(np.max(metal))
    return metrics


def pretty_print_metrics(metrics: dict):
    print("Invariant metrics:")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]:.6f}")


def test_vectorized(n: int = 2000, chunk: int = 1024) -> dict:
    print(f"[TEST] Generating {n} samples (chunk={chunk}) for invariant checks...")
    data = generate_data(n=n, chunk=chunk)
    metrics = run_invariants(data)
    pretty_print_metrics(metrics)
    # Simple assertions (tolerances arbitrary, adjust as needed)
    assert metrics['tangent_normal_dot_max_abs'] < 1e-4, "Tangent not orthogonal to normal within tolerance"
    assert metrics['shading_normal_len_max_dev'] < 1e-3, "Shading normal length deviation too large"
    assert metrics['pdf_min'] >= -1e-9, "PDF has negative values"
    return metrics

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Vectorized BSDF sample generator")
    ap.add_argument('--n', type=int, default=1024*16, help='Total number of samples')
    ap.add_argument('--chunk', type=int, default=16384, help='Batch size per vectorized iteration')
    ap.add_argument('--out', type=str, default='bsdf_rgb_samples_extended_vec.npz', help='Output .npz file')
    ap.add_argument('--test-only', action='store_true', help='Run invariants test & exit')
    ap.add_argument('--no-save', action='store_true', help='Do not persist the generated data')
    ap.add_argument('--compress', action='store_true', help='Save compressed .npz (default is uncompressed)')
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.test_only:
        test_vectorized(n=args.n, chunk=args.chunk)
        return

    print(f"Generating n={args.n} chunk={args.chunk} ...")
    data = generate_data(n=args.n, chunk=args.chunk)
    metrics = run_invariants(data)
    pretty_print_metrics(metrics)
    if not args.no_save:
        save_npz(data, args.out, compress=args.compress)


if __name__ == '__main__':
    main()