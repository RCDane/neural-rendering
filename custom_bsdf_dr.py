import os
import numpy as np
import mitsuba as mi
import drjit as dr  
# dr.set_flag(dr.JitFlag.SymbolicLoops, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
import drjit.auto.ad as drad
import json
# mi.set_variant("scalar_rgb")



class NeuralBSDF(mi.BSDF):
    """
    Python BSDF plugin that evaluates a trained SimpleNeuralBSDF.

    Scene usage example:
      {
        "type": "neural_bsdf",
        "albedo": [0.8, 0.7, 0.6],
        "roughness": 0.4,
        "metallic": 0.1,
        "weights": "checkpoints/best_model.pth",
        "hidden_dim": 32
      }

    Inputs to the network follow training:
      wi_local(3), wo_local(3), albedo(3), normal(3), roughness(1), metallic(1)
    """
    def __init__(self, props: mi.Properties):
        super().__init__(props)

        


        # Constant material parameters (could be extended to textures)
        self.base_color = props.get('base_color')
        self.roughness = props.get('roughness')
        self.metallic = props.get('metallic')
        self.normal_map = props.get('normal_map')

        self.input_dim = props.get('input_dim', 0)
        
        assert self.input_dim != 0, "[NeuralBSDF] 'input_dim' property must be specified."
        
        self.output_dim = props.get('output_dim', 3)
        
        # self.model_path = props.get('model_path', '')
        # assert self.model_path != '', "[NeuralBSDF] 'model_path' property must be specified."
        # metadata_path = self.model_path + '.json'
        
        # with open(metadata_path, 'r') as f:
        #     self.metadata = json.load(f) 
        
        # print("model metadata:", self.metadata)
        
        
        # # Network dimensions
        # self.input_dim = [item[1] for item in self.metadata['input_shape']]
        # self.hidden_dim = self.metadata["hidden_dim"]
        # self.output_dim = 3

        # if self.model_path:
        #     self._load_weights(self.model_path)
        # else:
        #     mi.Log(mi.LogLevel.Warn, "[NeuralBSDF] No model path provided; BSDF will output zeros.")

        # # Report supported flags
        self._flags = mi.BSDFFlags.Diffuse | mi.BSDFFlags.FrontSide

    def add_model(self, model: dr.nn.Module):
        self.model = model
    def flags(self) -> mi.BSDFFlags:
        return self._flags

    

    
    def eval(self, ctx: mi.BSDFContext,  si: mi.SurfaceInteraction3f,
             wo: mi.Vector3f, active=True) -> mi.Color3f:
        wi = si.wi
        # wi = si.to_local(wi)
        # wo = si.to_local(wo)
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        valid = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        h = dr.normalize(wi + wo)
        ndotl = cos_theta_i
        ndotv = cos_theta_o
        ndoth = h.z
        ldoth = dr.clamp(dr.dot(wi, h), 0.0, 1.0)

        # Evaluate (possibly textured) parameters
        base_color_val = drad.TensorXf16(self.base_color.eval(si))
        roughness_val = drad.TensorXf16(self.roughness.eval(si))
        metallic_val  = drad.TensorXf16(self.metallic.eval(si))
        wi_local = dr.reshape(drad.TensorXf16(wi),(3,-1))
        wo_local = dr.reshape(drad.TensorXf16(wo),(3,-1))
        # print("wi_local:", wi_local, "wi_local shape:", wi_local.shape)
        # print("wo_local:", wo_local, "wo_local shape:", wo_local.shape)
        # print("base_color_val:", base_color_val, "base_color_val shape:", base_color_val.shape)
        
        
        input_concat = dr.concat([wi_local, wo_local,
                                 metallic_val, roughness_val, base_color_val], axis=0)
        
        element_length = wi_local.shape[1]
        
        input = dr.nn.CoopVec(*input_concat)
        # input = dr.nn.CoopVec(wi_local, wo_local,
        #                          metallic_val, roughness_val, base_color_val)
        f_rgb = mi.Color3f(self.model(input))
        # print("f_rgb:",f_rgb, "f_rgb shape:", f_rgb.shape)

        return dr.select(active, f_rgb, mi.Color3f(0.0))
        


    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            wo: mi.Vector3f, active=True) -> mi.Float:
        cos_theta = mi.Frame3f.cos_theta(wo)
        inv_pi = 1.0 / np.pi
        return mi.Float(dr.select(cos_theta > 0, cos_theta * inv_pi, 0.0))

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
               sample1: mi.Float, sample2: mi.Point2f, active=True):
        # Cosine-weighted hemisphere using sample2
        u1 = sample2.x
        u2 = sample2.y
        r = dr.sqrt(u1)
        phi = 2.0 * dr.pi * u2
        x = r * dr.cos(phi)
        y = r * dr.sin(phi)
        z = dr.sqrt(dr.maximum(0.0, 1.0 - u1))
        wo = mi.Vector3f(x, y, z)

        f_val = self.eval(ctx, si, wo, active)
        cos_theta = mi.Frame3f.cos_theta(wo)
        pdf_val = dr.select(cos_theta > 0, cos_theta * (1.0 / np.pi), 0.0)

        bs = mi.BSDFSample3f()
        bs.wo = wo
        bs.pdf = pdf_val
        bs.sampled_type = self._flags & mi.BSDFFlags.Diffuse
        bs.sampled_component = 0
        bs.eta = 1.0

        weight = mi.Color3f(0.0)
        valid = active & (cos_theta > 0) & (pdf_val > 0)
        weight = mi.Color3f(dr.select(valid, f_val * (cos_theta / (pdf_val + 1e-8)), mi.Color3f(0.0)))
        return (bs, weight)

    def traverse(self, callback):
        callback.put_parameter("base_color", self.base_color)
        callback.put_parameter("roughness", self.roughness)
        callback.put_parameter("metallic", self.metallic)
        callback.put_parameter("normal_map", self.normal_map)

    def to_string(self):
        return "test 123"
        
mi.register_bsdf("neural_bsdf", NeuralBSDF)


