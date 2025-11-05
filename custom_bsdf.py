import os
import torch
import numpy as np
import mitsuba as mi
import drjit as dr  
# dr.set_flag(dr.JitFlag.SymbolicLoops, False)
# dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
import json
from model import SimpleNeuralBSDF, TorchToDRWrapper
# mi.set_variant("scalar_rgb")

def wrap_torch_module(module: torch.nn.Module,
                      target_dtype=None,
                      preserve_requires_grad: bool = False):
    """
    Wrap a torch.nn.Module so it can be called from Dr.Jit with
    automatic conversion + full forward/reverse AD interoperability.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to wrap. It can be on CPU or CUDA; inputs
        will be moved to its device automatically.
    target_dtype : optional
        A Dr.Jit array/tensor type you want outputs cast to. If None,
        the natural converted Dr.Jit types are returned.
        (E.g. dr.cuda.TensorXf, dr.llvm.TensorXf16, etc.)
    preserve_requires_grad : bool
        If True, torch inputs that require grad will keep that flag.
        Otherwise, they will be treated as leaf tensors without grad.

    Returns
    -------
    wrapped_fn : callable
        A function that takes Dr.Jit arrays (or PyTrees thereof) and
        returns Dr.Jit arrays. It participates in Dr.Jit autodiff:
        - reverse-mode (dr.backward)
        - forward-mode (dr.forward)
        - combined if needed.

    Usage
    -----
    torch_model = MyTorchNet().cuda()
    wrapped = wrap_torch_module(torch_model)

    # Dr.Jit side
    x = dr.cuda.TensorXf([1.0, 2.0, 3.0])
    dr.enable_grad(x)
    y = wrapped(x)
    dr.backward(y)
    grad_x = dr.grad(x)

    Notes
    -----
    - Only pass Dr.Jit arrays/tensors (or tuples/lists/dicts thereof) as inputs.
    - For forward-mode AD, PyTorch has a limitation when passing Python
      objects that are not tensors (see PyTorch issue #117491 referenced
      in Dr.Jit interop docs).
    - If you need multiple inputs, just call wrapped(a, b, c, ...).
    - Outputs can be tuples/dicts; they will map back into Dr.Jit PyTrees.
    """

    # Ensure module is in eval mode by default (you can change later).
    module.eval()

    # Capture device (CPU or CUDA) to move inputs there.
    device = next(module.parameters(), torch.tensor(0)).device

    @dr.wrap(source='drjit', target='torch') 
    def _wrapped(*args, **kwargs):
        torch_args = []
        for a in args:
            # a arrives as torch.Tensor already (due to interop), but ensure device/dtype alignment.
            if isinstance(a, torch.Tensor):
                t = a.to(device)
                if preserve_requires_grad and hasattr(a, 'requires_grad'):
                    t.requires_grad = a.requires_grad
                torch_args.append(t)
            else:
                torch_args.append(a)

        torch_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                tv = v.to(device)
                if preserve_requires_grad and hasattr(v, 'requires_grad'):
                    tv.requires_grad = v.requires_grad
                torch_kwargs[k] = tv
            else:
                torch_kwargs[k] = v

        out = module(*torch_args, **torch_kwargs)

        # Optionally cast outputs (after conversion back to Dr.Jit).
        # Interop will convert torch tensors → Dr.Jit arrays automatically;
        # we just post-process if target_dtype is specified.
        if target_dtype is None:
            return out

        def _cast_leaf(x):
            if dr.is_array_v(type(x)):
                # If already target_dtype, keep; else cast elementwise
                if type(x) is target_dtype:
                    return x
                # For Tensor target types, construct via target_dtype(x)
                try:
                    return target_dtype(x)
                except Exception:
                    # Fallback: create new array
                    return target_dtype([xi for xi in x])
            return x

        # Handle possible PyTree outputs
        if isinstance(out, (list, tuple)):
            return type(out)(_cast_leaf(o) for o in out)
        elif isinstance(out, dict):
            return {k: _cast_leaf(v) for k, v in out.items()}
        else:
            return _cast_leaf(out)

    return _wrapped




class NeuralBSDFEvaluator(dr.CustomOp):
    """
    This CustomOp safely wraps the PyTorch model call, making it compatible
    with Dr.Jit's symbolic execution and automatic differentiation.
    """
    
        # The input signature to the model is fixed based on the original code:
        # 4 scalars, 1 color (3 floats), 2 scalars = 9 total features
    def eval(self, model, ndotl, ndotv, ndoth, ldoth, base_color, roughness, metallic):
        """
        Defines the forward pass of the operation.
        This is called when using any variant of Mitsuba.
        """
        # while dr.hint(,mode='evaluated'):
        
        # Stack features: shape (N, 9)
        features = torch.tensor(
            [ndotl, ndotv, ndoth, ldoth,
            base_color.x, base_color.y, base_color.z, roughness, metallic]
        )

            # 3. Run the PyTorch model
        out_torch = model(features)

        # 4. Apply the same post-processing as in the original code
        processed_torch = torch.exp(out_torch - 3.0)

        # 5. Convert the [N, 3] PyTorch output tensor back to a Mitsuba Color3f

        return mi.Color3f(processed_torch)

    def backward(self, grad_output_dr):
        """
        Defines the backward pass for automatic differentiation.
        """
        # 1. Convert the incoming Dr.Jit gradient to a PyTorch tensor
        grad_output_torch = torch.cat([
            torch.utils.dlpack.from_dlpack(grad_output_dr.x.dlpack()).unsqueeze(1),
            torch.utils.dlpack.from_dlpack(grad_output_dr.y.dlpack()).unsqueeze(1),
            torch.utils.dlpack.from_dlpack(grad_output_dr.z.dlpack()).unsqueeze(1)
        ], dim=1)

        # 2. Perform the backward pass in PyTorch
        self.outputs_torch.backward(grad_output_torch)
        grad_input_torch = self.inputs_torch.grad

        # 3. Split the input gradient tensor and convert back to a tuple of Dr.Jit arrays,
        #    matching the signature of the 'eval' method's inputs.
        g_ndotl    = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 0].contiguous()))
        g_ndotv    = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 1].contiguous()))
        g_ndoth    = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 2].contiguous()))
        g_ldoth    = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 3].contiguous()))
        g_base_r   = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 4].contiguous()))
        g_base_g   = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 5].contiguous()))
        g_base_b   = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 6].contiguous()))
        g_rough    = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 7].contiguous()))
        g_metallic = dr.ArrayXf(torch.utils.dlpack.to_dlpack(grad_input_torch[:, 8].contiguous()))

        g_base_color = mi.Color3f(g_base_r, g_base_g, g_base_b)

        return (g_ndotl, g_ndotv, g_ndoth, g_ldoth, g_base_color, g_rough, g_metallic)



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

        self.model_path = props.get('model_path', '')
        assert self.model_path != '', "[NeuralBSDF] 'model_path' property must be specified."
        metadata_path = self.model_path + '.json'
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f) 
        
        print("model metadata:", self.metadata)
        
        
        # Network dimensions
        self.input_dim = [item[1] for item in self.metadata['input_shape']]
        self.hidden_dim = self.metadata["hidden_dim"]
        self.output_dim = 3

        if self.model_path:
            self._load_weights(self.model_path)
        else:
            mi.Log(mi.LogLevel.Warn, "[NeuralBSDF] No model path provided; BSDF will output zeros.")

        # Report supported flags
        self._flags = mi.BSDFFlags.Diffuse | mi.BSDFFlags.FrontSide

    def _load_weights(self, path: str):
        if not os.path.isfile(path):
            mi.Log(mi.LogLevel.Error, f"[NeuralBSDF] Weights file not found: {path}")
            return
        sd = torch.load(path)
        
        try:
            input_size = sd['model_state']['fc1.weight'].shape[1]
        except KeyError:
            mi.Log(mi.LogLevel.Error, f"[NeuralBSDF] Invalid state dict format in: {path}")
            return
        try:
            output_size = sd['model_state']['fc3.weight'].shape[0]
        except KeyError:
            mi.Log(mi.LogLevel.Error, f"[NeuralBSDF] Invalid state dict format in: {path}")
            return
        
        # device_str = 'cuda' if 'cuda' in mi.variant() else 'cpu'
        # print(f"[NeuralBSDF] Loading model on device: {device_str}")
        self.model = SimpleNeuralBSDF(input_size, self.hidden_dim, output_size)
        self.model.load_state_dict(sd['model_state'])
        self.wrapped_model = TorchToDRWrapper(self.model,3, target_dtype=dr.auto.ad.TensorXf16)
        # self.wrapped_model = wrap_torch_module(self.model, target_dtype=dr.auto.TensorXf)
        
        # self.model.eval()

        # Instantiate the custom operator that wraps our model
        # self.evaluator = NeuralBSDFEvaluator(self.model)

        if 'model_state' in sd:
            sd = sd['model_state']
        self.model.load_state_dict(sd, strict=True)
        mi.Log(mi.LogLevel.Info, f"[NeuralBSDF] Loaded weights from {path}")

    def flags(self) -> mi.BSDFFlags:
        return self._flags

    
    def run_wrapped_model(self, input):
        input = torch.tensor(input).unsqueeze(0)
        # print("input device", input.device())
        return self.model(input)
    
    def eval(self, ctx: mi.BSDFContext,  si: mi.SurfaceInteraction3f,
             wo: mi.Vector3f, active=True) -> mi.Color3f:
        wi = si.wi
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        valid = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        h = dr.normalize(wi + wo)
        ndotl = cos_theta_i
        ndotv = cos_theta_o
        ndoth = h.z
        ldoth = dr.clamp(dr.dot(wi, h), 0.0, 1.0)

        # Evaluate (possibly textured) parameters
        base_color_val = self.base_color.eval(si)
        roughness_val = self.roughness.eval(si)[0]
        metallic_val  = self.metallic.eval(si)[0]
        

        drinput = dr.auto.ad.TensorXf16([ndotl, ndotv, ndoth, ldoth,
                                 base_color_val.x, base_color_val.y, base_color_val.z,
                                 roughness_val, metallic_val])
        f_rgb = mi.Color3f(self.wrapped_model.eval(drinput))


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
        return (f"NeuralBSDF[base_color={self.base_color}, roughness={self.roughness}, "
                f"metallic={self.metallic}, hidden_dim={self.hidden_dim}, "
                f"model_path='{self.model_path}']")
# mi.set_variant('cuda_ad_rgb')  # CPU variant - more stable

mi.register_bsdf("neural_bsdf", NeuralBSDF)


# Register plugin name: "neural_bsdf"

# scene = {
# #   "type": "scene",
# #   "camera": { ... },
#   "shape": {
#      "type": "obj",
#      "filename": "data/lubricant_spray_8k.obj",
#      "bsdf": {
#         "type": "neural_bsdf",
#         "albedo": [0.9, 0.85, 0.8],
#         "roughness": 0.3,
#         "metallic": 0.05,
#         "weights": "checkpoints/best_model.pth",
#         "hidden_dim": 32
#      }
#   }
# }

# loaded_scene = mi.load_dict(scene)

# neuralBSDF = loaded_scene.shapes()[0].bsdf()

