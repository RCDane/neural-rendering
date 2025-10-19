import os
import torch
import numpy as np
import mitsuba as mi

from model import SimpleNeuralBSDF
mi.set_variant("scalar_rgb")


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
        self.albedo = mi.Color3f(props.get('albedo', mi.Color3f(0.8)))
        self.roughness = float(props.get('roughness', 0.5))
        self.metallic = float(props.get('metallic', 0.0))

        self.hidden_dim = int(props.get('hidden_dim', 32))
        self.weights_path = props.get('weights', '')

        # Network dimensions
        self.input_dim = 3 + 3 + 3 + 3 + 1 + 1
        self.output_dim = 3

        self.device = torch.device('cpu')
        self.model = SimpleNeuralBSDF(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.model.eval()

        if self.weights_path:
            self._load_weights(self.weights_path)
        else:
            mi.Log(mi.LogLevel.Warn, "[NeuralBSDF] No weights path provided; BSDF will output zeros.")

        # Report supported flags
        self._flags = mi.BSDFFlags.Diffuse | mi.BSDFFlags.FrontSide

    def _load_weights(self, path: str):
        if not os.path.isfile(path):
            mi.Log(mi.LogLevel.Error, f"[NeuralBSDF] Weights file not found: {path}")
            return
        sd = torch.load(path, map_location=self.device)
        if isinstance(sd, dict):
            if 'state_dict' in sd:
                sd = sd['state_dict']
            elif 'model_state' in sd:
                sd = sd['model_state']
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            mi.Log(mi.LogLevel.Warn, f"[NeuralBSDF] Missing keys: {missing}")
        if unexpected:
            mi.Log(mi.LogLevel.Warn, f"[NeuralBSDF] Unexpected keys: {unexpected}")
        mi.Log(mi.LogLevel.Info, f"[NeuralBSDF] Loaded weights from {path}")

    def flags(self) -> mi.BSDFFlags:
        return self._flags

    def eval(self, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, wi: mi.Vector3f, active=True) -> mi.Color3f:
        """
        Evaluate f(wi, wo) (no cosine factor) using the neural network.

        wo, wi: local directions (Mitsuba convention: +Z is the shading normal).
        """
        # Only front-side, ignore transmission
        cos_theta_i = wi.z
        cos_theta_o = wo.z
        valid = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        if not mi.any(valid):
            return mi.Color3f(0.0)

        # Assemble input tensor (batch size 1)
        wi_np = np.array([wi.x, wi.y, wi.z], dtype=np.float32)
        wo_np = np.array([wo.x, wo.y, wo.z], dtype=np.float32)
        n_np = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # local normal
        albedo_np = np.array([self.albedo.x, self.albedo.y, self.albedo.z], dtype=np.float32)
        roughness_np = np.array([self.roughness], dtype=np.float32)
        metallic_np = np.array([self.metallic], dtype=np.float32)

        inp = np.concatenate([wi_np, wo_np, albedo_np, n_np, roughness_np, metallic_np], axis=0)[None, :]
        inp_t = torch.from_numpy(inp).to(self.device)

        with torch.no_grad():
            raw = self.model(inp_t)[0].cpu().numpy()
            f_rgb = np.exp(raw - 3.0)  # same transform as training

        return mi.Color3f(*f_rgb) * mi.Color3f(valid)

    def pdf(self, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, wi: mi.Vector3f, active=True) -> mi.Float:
        """
        Cosine-weighted hemisphere PDF used in sample().
        """
        cos_theta = wi.z
        pdf = mi.Float(cos_theta * mi.InvPi) & active & (cos_theta > 0)
        return pdf

    def sample(self,
               mode: mi.BSDFFlags,
               si: mi.SurfaceInteraction3f,
               sample1: mi.Float,
               sample2: mi.Point2f,
               active=True):
        """
        Sample wi using cosine-weighted hemisphere. Return:
          (bsdf_weight, wi, pdf, flags)
        bsdf_weight = f(wi, wo) * cos_theta / pdf.
        """
        # wo is always (0,0,1) in local frame after shading frame transform
        # Generate cosine-weighted sample
        u1 = sample1
        u2 = sample2.x
        r = mi.Float(np.sqrt(u1))
        phi = mi.Float(2.0 * np.pi * u2)
        x = r * mi.cos(phi)
        y = r * mi.sin(phi)
        z = mi.Float(np.sqrt(mi.max(0.0, 1.0 - u1)))
        wi = mi.Vector3f(x, y, z)

        wo = mi.Vector3f(0.0, 0.0, 1.0)  # local outgoing direction

        f_val = self.eval(si, wo, wi, active)
        pdf_val = self.pdf(si, wo, wi, active)

        # Throughput weight
        weight = mi.Color3f(0.0)
        mask = (pdf_val > 0)
        weight = f_val * wi.z / mi.Color3f(pdf_val + 1e-8) * mi.Color3f(mask)

        return (weight, wi, pdf_val, self._flags & mi.BSDFFlags.Diffuse, active)

    def traverse(self, callback):
        # Expose optimizable parameters if desired
        callback.put_parameter("albedo", self.albedo)
        callback.put_parameter("roughness", mi.Float(self.roughness))
        callback.put_parameter("metallic", mi.Float(self.metallic))

    def to_string(self):
        return (f"NeuralBSDF[albedo={self.albedo}, roughness={self.roughness}, "
                f"metallic={self.metallic}, hidden_dim={self.hidden_dim}, "
                f"weights='{self.weights_path}']")



# Register plugin name: "neural_bsdf"
mi.register_bsdf("neural_bsdf", NeuralBSDF)

scene = {
#   "type": "scene",
#   "camera": { ... },
  "shape": {
     "type": "obj",
     "filename": "data/lubricant_spray_8k.obj",
     "bsdf": {
        "type": "neural_bsdf",
        "albedo": [0.9, 0.85, 0.8],
        "roughness": 0.3,
        "metallic": 0.05,
        "weights": "checkpoints/best_model.pth",
        "hidden_dim": 32
     }
  }
}

loaded_scene = mi.load_dict(scene)

neuralBSDF = loaded_scene.shapes()[0].bsdf()

