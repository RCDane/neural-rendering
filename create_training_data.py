import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")  # or "llvm_ad_rgb" / "cuda_ad_rgb" for vectorization
OBJ = "data/lubricant_spray_1k.obj"
TEX = "data/textures"
# Load mesh & create a scene or directly load a bsdf
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


scene = mi.load_dict(_scene_dict)
mesh : mi.Shape = scene.shapes()[0]

params = mi.traverse(scene)




faces = params.get("mesh.faces")
vertices = params.get("mesh.vertex_positions")
normals = params.get("mesh.vertex_normals")
uvs = params.get("mesh.vertex_texcoords")

def surface_area(v0, v1, v2):
    print(v0, v1, v2)
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = dr.cross(edge1, edge2)
    area = 0.5 * dr.norm(cross_prod, axis=-1)
    return area
print("vertices type:", type(vertices))
print("vertices: ", vertices)
print("faces type:", type(faces))
print("faces: ", faces)
face_count = int(dr.width(faces) / 3)
print("face count:", face_count)
vertex_array_ordered = mi.Vector3f(dr.gather(dr.auto.ad.Float, vertices, faces))

p0 : mi.Vector3f = mi.Vector3f(vertex_array_ordered[0])
print("p0:", p0)
# vertex_array_ordered = dr.reshape(vertex_array_ordered, (face_count, 3))
print("vertex_array_ordered shape:", dr.shape(vertex_array_ordered))
print("vertex_array_ordered:", vertex_array_ordered)
print("vertex example:", vertex_array_ordered[0])


counter = 0
while counter < face_count:
    x = vertex_array_ordered[counter*3 + 0]
    y = vertex_array_ordered[counter*3 + 1]
    z = vertex_array_ordered[counter*3 + 2]
    print("v0:", x, "v1:", y, "v2:", z)
    area = surface_area(x, y, z)
    counter += 1

loaded_faces = mi
print(faces)
print(vertices)
print(normals)
print(uvs)
# print(scene)
# bsdf = mesh.bsdf()
# N = dr.auto.ad.UInt(10)
# sampler : mi.Sampler = mi.load_dict({"type": "independent", "sample_count": 10})

# sample_uvs_from_shape(scene.meshes()[0], sampler, N)



# # generate N samples
# for i in range(5):
#     bsdf_val, spectrum, si, wo_world = calculate_sample(mesh, scene, sampler)
#     print("BSDF value shape:", bsdf_val)
#     print("si:", si)
#     print("wo_world:", wo_world)

# samples_2d = sampler.next_2d()          # shape (N, 2)
# pos_sample = mesh.sample_position(0.0, samples_2d)  # PositionSample3f
# si = mesh.eval_parameterization(pos_sample.uv, mi.RayFlags.All)

# bsdf = mesh.bsdf()

# bsdf_at_si = si.bsdf()

# print("width: ", dr.width(si.p))

# # 5. Example: cosine-weighted hemisphere sampling of outgoing direction
# #    (Demonstration only; adjust for your rendering/integration needs.)
# hemisphere_samples = sampler.next_2d()   # New random samples
# wo_local = mi.warp.square_to_cosine_hemisphere(hemisphere_samples)  # shape (N, 3)

# # Convert outgoing directions to world space (if needed)
# wo_world = si.to_world(wo_local)

# # 6. Evaluate / sample BSDF.
# # BSDF expects outgoing direction in local frame; si.wi is the incident direction.
# ctx = mi.BSDFContext()

# # (a) Direct evaluation of BSDF value for chosen wo_local directions:
# #     bsdf.eval returns the BSDF weight (includes cosine term depending on implementation).
# bsdf_val = bsdf_at_si.eval(ctx, si, wo_local)

# # (b) Proper BSDF sampling (generating wo from distribution):
# bsdf_samples_2d = sampler.next_2d()
# bsdf_samples1_2d = sampler.next_1d()
# bsdf_val, spectrum = bsdf_at_si.sample(ctx, si, bsdf_samples1_2d ,bsdf_samples_2d)

# # Convert sampled outgoing to world if needed:
# # sampled_wo_world = si.to_world()


# print("BSDF value shape:", bsdf_val)
# print("Sampled spectrum shape:", spectrum)
# # 7. Monte Carlo weights for position sampling: area-uniform
# #    Each sample's surface pdf is pos_sample.pdf (constant).
# #    If combining with BSDF sampling, a simple estimator might be:
# #    contrib = (emission_or_radiance * sample_val) / (pos_pdf * bsdf_pdf)
# pos_pdf = pos_sample.pdf          # scalar or length-N array
# bsdf_pdf = bsdf_val.pdf                    # from bsdf.sample


# # 8. Access geometric fields:
# positions = si.p
# normals_geo = si.n          # Geometric normal (unperturbed)
# uvs = si.uv

# # 9. (Optional) Show a few values
# print("First position:", positions[0])
# print("First geometric normal:", normals_geo[0])
# print("First UV:", uvs[0])
# print("First BSDF eval value:", bsdf_val)
# print("First sampled BSDF PDF:", bsdf_pdf)

# # 10. If you need per-sample albedo / base color texture value:
# #     Evaluate the base_color texture via underlying principled BSDF's parameters.
# #     Principled BSDF doesn't expose a direct 'base_color.eval' if its texture
# #     is embedded; you can approximate diffuse albedo by sampling bsdf.sample
# #     and averaging, or inspect internal params if exposed.
# #     For a quick approximate per-point diffuse reflectance:
# #     (This is a heuristic; correct decomposition depends on metallic/roughness.)
# diffuse_reflectance_est = bsdf_at_si.eval(ctx, si, mi.Vector3f(0,0,1))  # Using normal direction
# print("First diffuse-like estimate:", diffuse_reflectance_est[0])