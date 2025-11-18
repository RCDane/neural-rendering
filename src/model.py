import pickle
import render_neural_material
import mitsuba as mi



class NeuralBSDF(mi.BSDF):

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.network_path = props.string('network_path', '')
        self.weights = self._load_weights(self.network_path)

    def _load_weights(self, path: str):
        return None

    def flags(self) -> mi.BSDFFlags:
        return mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
    

        


    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            wo: mi.Vector3f, active=True) -> mi.Float:
        return mi.Float(0.0)

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
               sample1: mi.Float, sample2: mi.Point2f, active=True):
        return None, mi.Vector3f(0.0), mi.Float(0.0), mi.BSDFSample3f()

    def traverse(self, callback):
        return None

    def to_string(self):
        return "NeuralBSDF[]"