import mitsuba as mi
mi.set_variant('scalar_rgb')  # CPU variant - more stable
import drjit as dr
dr.set_flag(dr.JitFlag.SymbolicLoops, False)
dr.set_flag(dr.JitFlag.SymbolicConditionals, False)

# dr.set_backend(dr.JitBackend.)
dr.set_flag(dr.JitFlag.Debug, True)

import custom_bsdf

obj_file = "data/lubricant_spray_8k.obj"
texture_folder = "data/textures/"

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
            origin=[0, 0.1, 0.2], target=[0, 0.07, 0], up=[0, 1, 0]
        ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 400,
            'height': 400,
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
            'type': 'neural_bsdf',
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
            'normal_map': {
                'type': 'bitmap',
                'filename': texture_folder+'lubricant_spray_nor_gl_8k.exr',
                'raw': True            # normal maps are linear data
            },
            'model_path': 'checkpoints/run3/best_model.pth',

        }
        
    }
})

img = mi.render(scene, spp=16)
mi.util.write_bitmap('render.exr', img)
mi.util.write_bitmap('render.png', img)  # quick LDR preview