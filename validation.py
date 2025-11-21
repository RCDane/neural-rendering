import mitsuba as mi
import drjit as dr
import os
OBJ = "data/lubricant_spray_1k.obj"
TEX = "data/textures"


def validate_scene_principled(scene : mi.Scene,spp:int, output_path: os.path):
    """ Validate that a scene loaded from file has the expected structure
    # Ensure that all meshes have a BSDF assigned"""
    
    image = mi.render(scene, spp=spp)
    
    