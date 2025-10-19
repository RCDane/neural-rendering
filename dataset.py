import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------------------------------------------
# File loading helper
# ------------------------------------------------------------
def load_sample_file(path: str) -> dict[str, np.ndarray]:
    """Load a sample data file (.npz multi-array or structured .npy).

    .npz: returns mapping of contained arrays.
    .npy (structured dtype): returns mapping of named fields.
    Plain 2D .npy without field names is rejected (ambiguous layout).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path, mmap_mode='r')
        return {k: z[k] for k in z.files}
    if ext == ".npy":
        arr = np.load(path)
        if arr.dtype.names:
            return {n: arr[n] for n in arr.dtype.names}
        raise ValueError(".npy provided without structured dtype field names; cannot infer keys.")
    raise ValueError(f"Unsupported extension '{ext}'. Use .npz or structured .npy")


class NeuralBRDFDataset(Dataset):
    """Neural BRDF dataset loader (.npz or structured .npy).

    Required keys:
      pos, normal, uv, wi_local, wo_local, f, pdf,
      tangent, normal_map, shading_normal, albedo, roughness, metallic

    Derived features are computed on-the-fly (h, cos_i, cos_o, dot_wi_wo).
    """

    def __init__(self,
                 npz_path: str = "bsdf_rgb_samples_extended.npz",
                 return_dict: bool = True,
                 device: str | torch.device | None = None):
        super().__init__()
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        raw = load_sample_file(npz_path)
        self.return_dict = return_dict
        self.device = torch.device(device) if device else None

        def _req(key: str):
            if key not in raw:
                raise KeyError(f"Missing key '{key}' in {npz_path}")
            return raw[key]

        pos    = torch.from_numpy(_req("pos")).float()
        normal = torch.from_numpy(_req("normal")).float()
        uv     = torch.from_numpy(_req("uv")).float()
        wi     = torch.from_numpy(_req("wi_local")).float()
        wo     = torch.from_numpy(_req("wo_local")).float()
        f      = torch.from_numpy(_req("f")).float()
        pdf    = torch.from_numpy(_req("pdf")).float().unsqueeze(-1)

        tangent        = torch.from_numpy(_req("tangent")).float()
        normal_map     = torch.from_numpy(_req("normal_map")).float()
        shading_normal = torch.from_numpy(_req("shading_normal")).float()
        albedo         = torch.from_numpy(_req("albedo")).float()
        roughness_raw  = torch.from_numpy(_req("roughness")).float()
        metallic_raw   = torch.from_numpy(_req("metallic")).float()
        roughness = roughness_raw.unsqueeze(-1) if roughness_raw.ndim == 1 else roughness_raw
        metallic  = metallic_raw.unsqueeze(-1) if metallic_raw.ndim == 1 else metallic_raw

        h = wi + wo
        h = h / torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
        cos_i = wi[:, 2:3].clamp(-1, 1)
        cos_o = wo[:, 2:3].clamp(-1, 1)
        dot_wi_wo = (wi * wo).sum(-1, keepdim=True).clamp(-1, 1)

        # Store tensors
        self.pos = pos
        self.normal = normal
        self.uv = uv
        self.wi_local = wi
        self.wo_local = wo
        self.f = f
        self.pdf = pdf
        self.tangent = tangent
        self.normal_map = normal_map
        self.shading_normal = shading_normal
        self.albedo = albedo
        self.roughness = roughness
        self.metallic = metallic
        self.h = h
        self.cos_i = cos_i
        self.cos_o = cos_o
        self.dot_wi_wo = dot_wi_wo
        self.sample_weight = torch.ones_like(pdf)

        self.input_dim = 31
        self.target_dim = 3
        self.N = f.shape[0]

        if self.device:
            for name in ("pos","normal","uv","wi_local","wo_local","f","pdf","tangent",
                         "normal_map","shading_normal","albedo","roughness","metallic","h","cos_i",
                         "cos_o","dot_wi_wo","sample_weight"):
                setattr(self, name, getattr(self, name).to(self.device))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        return {
            "pos": self.pos[idx],
            "normal": self.normal[idx],
            "uv": self.uv[idx],
            "wi_local": self.wi_local[idx],
            "wo_local": self.wo_local[idx],
            "h": self.h[idx],
            "cos_i": self.cos_i[idx],
            "cos_o": self.cos_o[idx],
            "dot_wi_wo": self.dot_wi_wo[idx],
            "tangent": self.tangent[idx],
            "normal_map": self.normal_map[idx],
            "shading_normal": self.shading_normal[idx],
            "albedo": self.albedo[idx],
            "roughness": self.roughness[idx],
            "metallic": self.metallic[idx],
            "f": self.f[idx],
            "pdf": self.pdf[idx],
            "weight": self.sample_weight[idx],
        }


# ------------------------------------------------------------
# Helper to create train/val loaders
# ------------------------------------------------------------
def create_dataloaders(path: str = "bsdf_rgb_samples_extended.npz",
                       batch_size: int = 1024,
                       val_ratio: float = 0.1,
                       num_workers: int = 0,
                       shuffle_train: bool = True,
                       pin_memory: bool = True,
                       **dataset_kwargs):
    full_ds = NeuralBRDFDataset(npz_path=path, **dataset_kwargs)
    val_size = int(len(full_ds) * val_ratio)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=shuffle_train, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=False, persistent_workers=bool(num_workers))
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=False, persistent_workers=bool(num_workers))
    return train_loader, val_loader, full_ds


# ------------------------------------------------------------
# Minimal usage example
# ------------------------------------------------------------
if __name__ == "__main__":
    ds = NeuralBRDFDataset(npz_path="bsdf_rgb_samples_extended.npz")
    print("Samples:", len(ds))
    first = ds[0]
    print("Keys:", list(first.keys()))
    print("Logical input_dim:", ds.input_dim, "target_dim:", ds.target_dim)
    # Try structured .npy if present
    alt = "bsdf_rgb_samples_struct.npy"
    if os.path.isfile(alt):
        try:
            ds2 = NeuralBRDFDataset(npz_path=alt)
            print("Structured .npy samples:", len(ds2))
        except Exception as e:
            print("Failed loading structured .npy:", e)