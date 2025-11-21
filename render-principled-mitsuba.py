import mitsuba as mi


obj_file = "data/lubricant_spray_1k.obj"
texture_folder = "data/textures/"
mi.set_variant('llvm_ad_rgb')  # CPU variant - more stable

scene = mi.load_dict({
    'type': 'scene',

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
				'width': 1024,
				'height': 1024,
				'rfilter': {'type': 'box'}
			}
		},

    # Your mesh with Principled BSDF
    'mesh': {
        'type': 'obj',
        'filename': obj_file,
        'face_normals': True,          # good default for OBJ
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
                # Optional tweaks:
                # 'specular': 0.5,
                # 'spec_tint': 0.0,
                # 'clearcoat': 0.0,
                # 'clearcoat_roughness': 0.03,
                # 'anisotropic': 0.0,
            }
        }
    }
})

integrator = mi.load_dict({ 'type': 'path' })


img = mi.render(scene, spp=512, integrator=integrator)
mi.util.write_bitmap('render1.exr', img)
mi.util.write_bitmap('render1.png', img)  # quick LDR preview