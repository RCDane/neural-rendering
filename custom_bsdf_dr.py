import os
import numpy as np
import mitsuba as mi
import drjit as dr  
# dr.set_flag(dr.JitFlag.SymbolicLoops, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
import drjit.auto.ad as drad
import json

from sympy import denom
import sampling
# mi.set_variant("scalar_rgb")

dr.syntax
def sigmoid(x):
    return 1 / (1 + dr.exp(-x))

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

    def add_model(self, 
                  model: dr.nn.Module, 
                  encoder_model: dr.nn.Module = None,
                  shading_frame_model: dr.nn.Module = None,
                  importance_sampling_model: dr.nn.Module = None):
        self.model = model
        self.encoder_model = encoder_model
        self.shading_frame_model = shading_frame_model
        self.importance_sampling_model = importance_sampling_model
    
    def add_texture(self, latent_texture: drad.Texture2f16):
        self.latent_texture = latent_texture
    def flags(self) -> mi.BSDFFlags:
        return self._flags

    
    def run_shading_frame_model(self, input, wo, wi):
        
        
        if self.shading_frame_model is None:
            return wi, wo
        coopVec_input = dr.nn.CoopVec(input)
        
        pred_fram = drad.TensorXf(self.shading_frame_model(coopVec_input))
            
        sh1_n = drad.Array3f(pred_fram[0:3])
        sh1_t = drad.Array3f(pred_fram[3:6])
        sh2_n = drad.Array3f(pred_fram[6:9])
        sh2_t = drad.Array3f(pred_fram[9:12])
        
        sh1_b = dr.cross(sh1_n, sh1_t)
        sh2_b = dr.cross(sh2_n, sh2_t)
        
        shading_frame1 = drad.Matrix3f16(sh1_n, sh1_t, sh1_b)
        shading_frame2 = drad.Matrix3f16(sh2_n, sh2_t, sh2_b)
        
        
        
        wi_transformed = dr.matmul(shading_frame1, drad.Array3f16(wi))
        wo_transformed = dr.matmul(shading_frame2, drad.Array3f16(wo))
        
        wi_local = drad.TensorXf16(wi_transformed)
        wo_local = drad.TensorXf16(wo_transformed)
        
        return wi_local, wo_local

    
    def eval(self, ctx: mi.BSDFContext,  si: mi.SurfaceInteraction3f,
             wo: mi.Vector3f, active=True) -> mi.Color3f:
        wi = si.wi
        wi = si.to_local(wi)
        wo = si.to_local(wo)
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        valid = active & (cos_theta_i > 0) & (cos_theta_o > 0)



        # Evaluate (possibly textured) parameters
        
        base_color = self.base_color.eval(si)
        normal_map = self.normal_map.eval(si)
        roughness = self.roughness.eval_1(si)
        metallic  = self.metallic.eval_1(si)
        
        dr.eval(base_color, roughness, metallic, normal_map)
        roughness_val = dr.reshape(drad.TensorXf16(roughness), (1, -1))
        metallic_val  = dr.reshape(drad.TensorXf16(metallic), (1, -1))
        wi_local = dr.reshape(drad.TensorXf16(wi),(3,-1))
        wo_local = dr.reshape(drad.TensorXf16(wo),(3,-1))
        
        
            
        
        base_color_val = drad.TensorXf16(base_color)
        normal_val = drad.TensorXf16(normal_map)
        
        coop_vector_input = dr.nn.CoopVec(metallic_val, roughness_val, *drad.TensorXf16(base_color), *normal_val)
        uses_latent = False
        
        if self.encoder_model is not None and self.latent_texture is None:
            encoded_texel = self.encoder_model(coop_vector_input)
            texel_tensor = drad.TensorXf16(encoded_texel)
            
            wi, wo = self.run_shading_frame_model(texel_tensor, wi_local, wo_local)
            
            input_concat = dr.concat([wi, wo,
                                    texel_tensor], axis=0)
            input = dr.nn.CoopVec(*input_concat)
            uses_latent = True
        elif self.latent_texture is not None:
            latent_vals = self.latent_texture.eval(si.uv)
            
            wi, wo = self.run_shading_frame_model(drad.TensorXf16(latent_vals), wi_local, wo_local)
            
            input_concat = dr.concat([wi, wo,
                                    drad.TensorXf16(latent_vals)], axis=0)
            input = dr.nn.CoopVec(*input_concat)
            uses_latent = True
        else:
            
            wi, wo = self.run_shading_frame_model(coop_vector_input, wi_local, wo_local)
            
            input_concat = dr.concat([wi, wo,
                                    metallic_val, roughness_val, base_color_val], axis=0)
            input = dr.nn.CoopVec(*input_concat)
            uses_latent = False
            
        
            
        # input = dr.nn.CoopVec(wi_local, wo_local,
        #                          metallic_val, roughness_val, base_color_val)
        
        output = drad.TensorXf(self.model(input))
        
        
        
        f_rgb = mi.Color3f(output[:3])
        
        # print("f_rgb:",f_rgb, "f_rgb shape:", f_rgb.shape)

        return dr.select(active, f_rgb, mi.Color3f(0.0))
        


    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            wo: mi.Vector3f, active=True) -> mi.Float:
        cos_theta = mi.Frame3f.cos_theta(wo)
        inv_pi = 1.0 / np.pi
        return mi.Float(dr.select(cos_theta > 0, 1.0, 0.0))
    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
               sample1: mi.Float, sample2: mi.Point2f, active=True):
        # Cosine-weighted hemisphere using sample2
        
        if self.encoder_model is not None and self.importance_sampling_model is not None:
            
            if self.latent_texture is None:
                base_color = self.base_color.eval(si)
                normal_map = self.normal_map.eval(si)
                roughness = self.roughness.eval_1(si)
                metallic  = self.metallic.eval_1(si)
                
                dr.eval(base_color, roughness, metallic, normal_map)
                roughness_val = dr.reshape(drad.TensorXf16(roughness), (1, -1))
                metallic_val  = dr.reshape(drad.TensorXf16(metallic), (1, -1))
                wi_local = dr.reshape(drad.TensorXf16(si.wi),(3,-1))
                normal_val = drad.TensorXf16(normal_map)
                encoded_texel = self.encoder_model(dr.nn.CoopVec(metallic_val, roughness_val, *drad.TensorXf16(base_color), *normal_val))
            else:
                latent_vals = self.latent_texture.eval(si.uv)
                wi_local = dr.reshape(drad.TensorXf16(si.wi),(3,-1))
                encoded_texel = drad.TensorXf16(latent_vals)
            importance_input = dr.nn.CoopVec(*wi_local, *encoded_texel)
            params = self.importance_sampling_model(importance_input)
            decoded = drad.TensorXf(params)

            alpha = mi.Vector3f(dr.exp(decoded[:3]-3.0))
            slope_spec = mi.Vector2f(dr.sinh(decoded[3:5]))
            slope_diff = mi.Vector2f(dr.sinh(decoded[5:7]))
            weight_spec = drad.Float(sigmoid(decoded[7]))

            wo_spec, wo_diff, pdf_spec, pdf_diff, pdf_pred = sampling.sample_analytic(
                drad.Array3f16(alpha), drad.Array2f16(slope_spec), drad.Array2f16(slope_diff), drad.Float16(weight_spec),
                drad.Array3f16(wi_local), drad.Array2f16(sample2)
            )
            
            is_specular = sample1 < weight_spec
            wo = dr.select(is_specular, wo_spec, wo_diff)
            pdf = dr.select(is_specular, pdf_spec, pdf_diff)
        else:     
            u1 = sample2.x
            u2 = sample2.y
            r = dr.sqrt(u1)
            phi = 2.0 * dr.pi * u2
            x = r * dr.cos(phi)
            y = r * dr.sin(phi)
            z = dr.sqrt(dr.maximum(0.0, 1.0 - u1))
            wo = mi.Vector3f(x, y, z)
            pdf = 1.0

        f_val = self.eval(ctx, si, wo, active) / pdf
        cos_theta = mi.Frame3f.cos_theta(wo)
        # pdf_val = dr.select(cos_theta > 0, 1.0, 0.0)

        bs = mi.BSDFSample3f()
        bs.wo = wo
        bs.pdf = dr.select(active, pdf, 0.0)
        bs.sampled_type = self._flags & mi.BSDFFlags.Diffuse
        bs.sampled_component = 0
        bs.eta = 1.0
        denom = dr.maximum(pdf, 1e-6)

        weight = mi.Color3f(0.0)
        valid = active & (cos_theta > 0) & (pdf > 0)
        weight = (f_val / denom) & valid
        return (bs, weight)

    def traverse(self, callback):
        callback.put_parameter("base_color", self.base_color)
        callback.put_parameter("roughness", self.roughness)
        callback.put_parameter("metallic", self.metallic)
        callback.put_parameter("normal_map", self.normal_map)

    def to_string(self):
        return "test 123"
        
mi.register_bsdf("neural_bsdf", NeuralBSDF)


