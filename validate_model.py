import os
import argparse
import torch
import mitsuba as mi
import numpy as np

from model import SimpleNeuralBSDF


def load_model(weights_path: str, input_dim: int, hidden_dim: int, output_dim: int):
    device = torch.device('cpu')
    model = SimpleNeuralBSDF(input_dim, hidden_dim, output_dim).to(device)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        elif 'model_state' in sd:        # accept training.py legacy key
            sd = sd['model_state']
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"[load_model] Missing keys: {missing}")
    if unexpected:
        print(f"[load_model] Unexpected keys: {unexpected}")
    model.eval()
    return model


class TextureSet:
	"""Container for material textures (albedo, roughness, metallic).

	Each texture is stored as a numpy array with shape (H, W, C). Roughness / metallic
	can be single channel; we will broadcast as needed.
	"""
	def __init__(self, albedo=None, roughness=None, metallic=None):
		self.albedo = albedo
		self.roughness = roughness
		self.metallic = metallic


def _load_bitmap(path: str, raw: bool = False):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Texture not found: {path}")
    bmp = mi.Bitmap(path)
    arr = np.array(bmp, copy=True)
    # Mitsuba returns (H, W) for single-channel; ensure (H, W, C)
    if arr.ndim == 2:
        arr = arr[..., None]
    if not raw:
        arr = np.power(np.clip(arr, 0, 1), 2.2)
    return arr


def load_textures(folder: str) -> TextureSet:
	"""Load required textures from a folder.

	Expected naming (can adapt):
	  - albedo: lubricant_spray_diff_8k.jpg (or png)
	  - roughness: lubricant_spray_rough_8k.exr
	  - metallic: lubricant_spray_metal_8k.exr
	"""
	# Attempt both jpg and png for albedo
	albedo_path_jpg = os.path.join(folder, 'lubricant_spray_diff_8k.jpg')
	albedo_path_png = os.path.join(folder, 'lubricant_spray_diff_8k.png')
	if os.path.isfile(albedo_path_jpg):
		albedo = _load_bitmap(albedo_path_jpg, raw=False)
	elif os.path.isfile(albedo_path_png):
		albedo = _load_bitmap(albedo_path_png, raw=False)
	else:
		raise FileNotFoundError("Albedo texture (diff) not found (jpg/png).")

	roughness = _load_bitmap(os.path.join(folder, 'lubricant_spray_rough_8k.exr'), raw=True)
	metallic = _load_bitmap(os.path.join(folder, 'lubricant_spray_metal_8k.exr'), raw=True)

	# Ensure single channel for roughness/metallic
	if roughness.shape[-1] > 1:
		roughness = roughness[..., :1]
	if metallic.shape[-1] > 1:
		metallic = metallic[..., :1]

	return TextureSet(albedo=albedo, roughness=roughness, metallic=metallic)


def sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
	"""Sample texture (nearest) given uv in [0,1].

	uv: (N, 2)
	return: (N, C)
	"""
	H, W, C = tex.shape
	# Wrap UVs
	u = (uv[:, 0] % 1.0) * (W - 1)
	v = (uv[:, 1] % 1.0) * (H - 1)
	u = np.clip(np.round(u).astype(int), 0, W - 1)
	v = np.clip(np.round(v).astype(int), 0, H - 1)
	return tex[v, u, :]


def bilinear_sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sampling for smoother results."""
    # Accept (H, W) by expanding to (H, W, 1)
    if tex.ndim == 2:
        tex = tex[..., None]
    H, W, C = tex.shape
    u = (uv[:, 0] % 1.0) * (W - 1)
    v = (uv[:, 1] % 1.0) * (H - 1)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = u - u0
    dv = v - v0
    c00 = tex[v0, u0, :]
    c10 = tex[v0, u1, :]
    c01 = tex[v1, u0, :]
    c11 = tex[v1, u1, :]
    c0 = c00 * (1 - du[:, None]) + c10 * du[:, None]
    c1 = c01 * (1 - du[:, None]) + c11 * du[:, None]
    return c0 * (1 - dv[:, None]) + c1 * dv[:, None]


def main():
	parser = argparse.ArgumentParser(description="Validate neural BSDF model by custom CPU-based rendering integration.")
	parser.add_argument('--weights', required=True, help='Path to trained model weights (.pt or .pth).')
	parser.add_argument('--obj', default='data/lubricant_spray_8k.obj', help='OBJ mesh path.')
	parser.add_argument('--textures', default='data/textures', help='Folder containing material textures.')
	parser.add_argument('--width', type=int, default=256, help='Render width.')
	parser.add_argument('--height', type=int, default=256, help='Render height.')
	parser.add_argument('--spp', type=int, default=64, help='Samples per pixel (hemisphere samples for integration).')
	parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension used in training.')
	args = parser.parse_args()

	# Match training input dimensions: wi(3)+wo(3)+albedo(3)+normal(3)+roughness(1)+metallic(1)
	input_dim = 3 + 3 + 3 + 3 + 1 + 1
	output_dim = 3
	model = load_model(args.weights, input_dim, args.hidden_dim, output_dim)
	print("Loaded model on CPU.")

	# Set Mitsuba variant (scalar for CPU)
	mi.set_variant('scalar_rgb')

	# Build scene with mesh only (no builtin BSDF shading, we sample ourselves)
	scene_dict = {
		'type': 'scene',
		'camera': {
			'type': 'perspective',
			'to_world': mi.ScalarTransform4f.look_at(
				origin=[0.4, 0.4, 0.4], target=[0, 0, 0], up=[0, 1, 0]
			),
			'fov': 45,
			'film': {
				'type': 'hdrfilm',
				'width': args.width,
				'height': args.height,
				'rfilter': {'type': 'box'}
			}
		},
		'mesh': {
			'type': 'obj',
			'filename': args.obj,
			'face_normals': True
		}
	}
	scene = mi.load_dict(scene_dict)

	textures = load_textures(args.textures)
	print("Textures loaded.")

	# Generate primary rays
	film_size = (args.height, args.width)
	xs = np.linspace(0.0, 1.0, args.width, endpoint=False) + 0.5/args.width
	ys = np.linspace(0.0, 1.0, args.height, endpoint=False) + 0.5/args.height
	uu, vv = np.meshgrid(xs, ys)
	pixel_uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)

	# Mitsuba camera sampling: use sample_ray with time=0, wavelength sample ignored (RGB)
	sensor = scene.sensors()[0]
	# Convert pixel coordinates to sample positions
	# sample_ray expects (time, sample2, sample2) for position & aperture; we supply pixel center.
	rays = []
	for uv in pixel_uv:
		wavelength_sample = 0.5
		film_sample = mi.Point2f(uv[0], uv[1])
		aperture_sample = mi.Point2f(0.5, 0.5)
		ray, _ = sensor.sample_ray(
            0.0,
            wavelength_sample,
            film_sample,
            aperture_sample
		)
		rays.append(ray)
	# Intersect rays
	its_list = [scene.ray_intersect(r) for r in rays]

	# Prepare output buffer
	spp = args.spp
	env_radiance = np.array([1.0,1.0,1.0])  # constant env
	out_rgb = np.zeros((len(its_list), 3), dtype=np.float32)

	# For each intersection, integrate neural BSDF * Li over hemisphere (Monte Carlo cosine-weighted)
	for i, its in enumerate(its_list):
		if not its.is_valid():
			out_rgb[i,:] = 0.0
			continue
		# Fetch shading data
		n = np.array(its.sh_frame.n, dtype=np.float32)
		t = np.array(its.sh_frame.s, dtype=np.float32)  # tangent
		b = np.array(its.sh_frame.t, dtype=np.float32)  # bitangent
		wo_world = -np.array(rays[i].d, dtype=np.float32)
		wo_world /= np.linalg.norm(wo_world) + 1e-8
		# Transform wo to local frame (n,t,b) where n is z-axis
		T = np.stack([t, b, n], axis=1)  # world = T * local
		# local = T^T * world
		wo_local = T.T @ wo_world

		# UV for textures
		uv = np.array(its.uv, dtype=np.float32)[None,:]
		albedo = bilinear_sample_texture(textures.albedo, uv)[0]  # (3,)
		roughness = bilinear_sample_texture(textures.roughness, uv)[0,0:1]  # (1,)
		metallic = bilinear_sample_texture(textures.metallic, uv)[0,0:1]  # (1,)

		# Monte Carlo integration over hemisphere
		accum = np.zeros(3, dtype=np.float32)
		for s in range(spp):
			# Cosine-weighted hemisphere sample
			u1, u2 = np.random.rand(), np.random.rand()
			r = np.sqrt(u1)
			theta = 2 * np.pi * u2
			x = r * np.cos(theta)
			y = r * np.sin(theta)
			z = np.sqrt(max(0.0, 1.0 - u1))
			wi_local = np.array([x, y, z], dtype=np.float32)
			# Convert back to world if needed (for env); env constant so skip direction query

			# Assemble model input
			# Ensure shapes (1, dim)
			inp = np.concatenate([
				wi_local,               # 3
				wo_local,               # 3
				albedo,                 # 3
				n,                      # 3 (using shading normal)
				roughness,              # 1
				metallic                # 1
			], axis=0)[None,:]
			inp_t = torch.from_numpy(inp).float()
			with torch.no_grad():
				f_rgb = model(inp_t).cpu().numpy()[0]
				f_rgb = np.exp(f_rgb - 3.0)  # same output transform as training
			cos_theta = wi_local[2]  # in local frame normal is (0,0,1)
			if cos_theta > 0:
				Li = env_radiance  # constant environment
				# Importance sampling pdf for cosine-weighted hemisphere: pdf = cos_theta / pi
				pdf = cos_theta / np.pi
				contrib = f_rgb * Li * cos_theta / (pdf + 1e-8)
				accum += contrib
		accum /= spp
		out_rgb[i,:] = accum

	# Reshape to image
	img = out_rgb.reshape(film_size[0], film_size[1], 3)
	# Clamp for safety
	img = np.clip(img, 0, 10.0)
	# Write images
	mi.util.write_bitmap('neural_render.exr', mi.Bitmap(img))
	mi.util.write_bitmap('neural_render.png', mi.Bitmap(img))
	print("Neural render saved to neural_render.exr / neural_render.png")


if __name__ == '__main__':
	main()
