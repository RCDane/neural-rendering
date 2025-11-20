import mitsuba as mi
import drjit as dr
OBJ = "data/lubricant_spray_1k.obj"
TEX = "data/textures"

scene_dict = {
		'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'env': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
        },
		'camera': {
			'type': 'perspective',
			'to_world': mi.ScalarTransform4f.look_at(
				origin=[0.2, 0.2, 0.2], target=[0, 0, 0], up=[0, 1, 0]
			),
			'fov': 45,
			'film': {
				'type': 'hdrfilm',
				'width': 400,
				'height': 400,
				'rfilter': {'type': 'box'}
			}
		},
		'mesh': 
        #     {

        #         'type': 'obj',
        #         'filename': OBJ,
        #         'face_normals': True,
        #         'bsdf': {
        #             'type': 'normalmap',
        #             'normalmap': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_nor_gl_1k.exr', 'raw': True},
        #             'bsdf': {
        #                 'type': 'principled',
        #                 'base_color': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_diff_1k.jpg', 'raw': False},
        #                 'metallic':   {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_metal_1k.exr', 'raw': True},
        #                 'roughness':  {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_rough_1k.exr', 'raw': True},
        #             }
        #         }
        # }
      {
			'type': 'obj',
            'filename': OBJ,
            'face_normals': True,
            'bsdf': None
		}
	}

principled_bsdf = {
            'type': 'normalmap',
            'normalmap': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_nor_gl_1k.exr', 'raw': True},
            'bsdf': {
                'type': 'principled',
                'base_color': {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_diff_1k.jpg', 'raw': False},
                'metallic':   {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_metal_1k.exr', 'raw': True},
                'roughness':  {'type': 'bitmap', 'filename': f'{TEX}/lubricant_spray_rough_1k.exr', 'raw': True},
            }
        }


def validate_scene_principled(scene_dict):
    # Ensure that all meshes have a BSDF assigned
    
    