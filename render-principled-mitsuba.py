import mitsuba as mi
import drjit as dr

obj_file = "data/lubricant_spray_1k.obj"
texture_folder = "data/textures/"
mi.set_variant('cuda_ad_rgb')  # CPU variant - more stable

scene = mi.load_dict({
    'type': 'scene',

        'env': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
        },
		'camera': {
			'type': 'perspective',
			'to_world': mi.ScalarTransform4f.look_at(
				origin=[0.2, 0.2, 0.2], target=[0, 0.1, 0], up=[0, 1, 0]
			),
			'fov': 30.0,
			'film': {
				'type': 'hdrfilm',
				'width': 1024,
				'height': 1024,
				'rfilter': {'type': 'box'}
			}
		},

    # Your mesh with Principled BSDF
    'mesh': {
        'type': 'obj',
        'filename': obj_file,
        'face_normals': False,          # good default for OBJ
        # Wrap the Principled BSDF with a normalmap BSDF
        'bsdf': {
            'type': 'normalmap',
            'normalmap': {
                'type': 'bitmap',
                'filename': texture_folder+'lubricant_spray_nor_gl_1k.exr',
                'raw': True            # normal maps are linear data
            },
            'bsdf': {
                'type': 'principled',
                'base_color': {
                    'type': 'bitmap',
                    'filename': texture_folder+'lubricant_spray_diff_1k.jpg',
                    'raw': False       # sRGB for color textures
                },
                'metallic': {
                    'type': 'bitmap',
                    'filename': texture_folder+'lubricant_spray_metal_1k.exr',
                    'raw': True
                },
                'roughness': {
                    'type': 'bitmap',
                    'filename': texture_folder+'lubricant_spray_rough_1k.exr',
                    'raw': True
                },
            }
        }
    }
})

integrator = mi.load_dict({ 'type': 'path' })
dr.set_flag(dr.JitFlag.SymbolicCalls, False)
dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
dr.set_flag(dr.JitFlag.SymbolicLoops, False)
import time
img = mi.render(scene, spp=32, integrator=integrator)
dr.sync_thread()

start_time = time.time()

img = mi.render(scene, spp=32, integrator=integrator)
dr.sync_thread()
end_time = time.time()
print(f"Render time for 64 samples: {end_time - start_time:.2f} seconds")
mi.util.write_bitmap('render2.exr', img)
mi.util.write_bitmap('render2.png', img)  # quick LDR preview