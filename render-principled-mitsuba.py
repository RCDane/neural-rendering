import mitsuba as mi


obj_file = "data/lubricant_spray_8k.obj"
texture_folder = "data/textures/"
mi.set_variant('scalar_rgb')  # CPU variant - more stable

scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path', 'max_depth': 8},

    # Environment light (replace with an HDRI if you like)
    'env': {
        'type': 'constant',
        'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
    },

    # Camera
    'camera': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0.4, 0.4, 0.4], target=[0, 0, 0], up=[0, 1, 0]
        ),
        'fov': 45,
        'film': {
            'type': 'hdrfilm',
            'width': 1080,
            'height': 1080,
            'rfilter': {'type': 'gaussian'}
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
                'filename': texture_folder+'lubricant_spray_nor_gl_8k.exr',
                'raw': True            # normal maps are linear data
            },
            'bsdf': {
                'type': 'principled',
                'base_color': {
                    'type': 'bitmap',
                    'filename': texture_folder+'lubricant_spray_diff_8k.jpg',
                    'raw': False       # sRGB for color textures
                },
                'metallic': {
                    'type': 'bitmap',
                    'filename': texture_folder+'lubricant_spray_metal_8k.exr',
                    'raw': True
                },
                'roughness': {
                    'type': 'bitmap',
                    'filename': texture_folder+'lubricant_spray_rough_8k.exr',
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

img = mi.render(scene, spp=256)
mi.util.write_bitmap('render.exr', img)
mi.util.write_bitmap('render.png', img)  # quick LDR preview