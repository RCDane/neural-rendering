import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Union, List

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


class NeuralBRDFDatasetOptimized(Dataset):
    """Optimized Neural BRDF dataset loader with batch-aware access patterns.

    Required keys:
      pos, normal, uv, wi_local, wo_local, f, pdf,
      tangent, normal_map, shading_normal, albedo, roughness, metallic

    Optimizations:
    1. Support for batch indexing via slices and lists
    2. Pre-computed derived features stored efficiently
    3. Minimal memory overhead per access
    4. Optional in-memory caching for frequently accessed ranges
    """

    def __init__(self,
                 npz_path: str = "bsdf_samples.npz",
                 return_dict: bool = True,
                 cache_size: int = 0,  # Number of samples to cache in memory (0 = no cache)
                 preload_features: bool = True):  # Precompute derived features
        super().__init__()
        self.device = None
        self.cache_size = cache_size
        self.preload_features = preload_features
        
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        raw = load_sample_file(npz_path)
        self.return_dict = return_dict

        def _req(key: str):
            if key not in raw:
                raise KeyError(f"Missing key '{key}' in {npz_path}")
            return raw[key]

        # Load all required tensors
        self._load_tensors(raw, _req)
        
        # Set up cache if requested
        self.cache = {} if cache_size > 0 else None
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_tensors(self, raw, _req):
        """Load and convert tensors from raw data."""
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

        # Pre-compute derived features if requested
        if self.preload_features:
            h = wi + wo
            h = h / torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
            cos_i = wi[:, 2:3].clamp(-1, 1)
            cos_o = wo[:, 2:3].clamp(-1, 1)
            dot_wi_wo = (wi * wo).sum(-1, keepdim=True).clamp(-1, 1)
        else:
            h = cos_i = cos_o = dot_wi_wo = None

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

    def _compute_derived_features(self, wi, wo):
        """Compute derived features on-the-fly if not preloaded."""
        h = wi + wo
        h = h / torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
        cos_i = wi[:, 2:3].clamp(-1, 1)
        cos_o = wo[:, 2:3].clamp(-1, 1)
        dot_wi_wo = (wi * wo).sum(-1, keepdim=True).clamp(-1, 1)
        return h, cos_i, cos_o, dot_wi_wo

    def to_device(self, device):
        """Move all tensors to specified device."""
        self.device = device
        for name in ("pos","normal","uv","wi_local","wo_local","f","pdf","tangent",
                     "normal_map","shading_normal","albedo","roughness","metallic",
                     "sample_weight"):
            setattr(self, name, getattr(self, name).to(device))
        
        if self.preload_features:
            for name in ("h","cos_i","cos_o","dot_wi_wo"):
                if getattr(self, name) is not None:
                    setattr(self, name, getattr(self, name).to(device))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: Union[int, slice, List[int]]):
        """Support both single indexing and batch indexing."""
        
        # Handle different index types
        if isinstance(idx, int):
            return self._get_single_item(idx)
        elif isinstance(idx, slice):
            return self._get_slice_items(idx)
        elif isinstance(idx, (list, tuple, torch.Tensor)):
            return self._get_list_items(idx)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def _get_single_item(self, idx: int):
        """Get a single item (original behavior)."""
        # Check cache first
        if self.cache is not None and idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]
        
        self.cache_misses += 1
        
        # Get derived features
        if self.preload_features:
            h, cos_i, cos_o, dot_wi_wo = (self.h[idx], self.cos_i[idx], 
                                         self.cos_o[idx], self.dot_wi_wo[idx])
        else:
            wi = self.wi_local[idx:idx+1]
            wo = self.wo_local[idx:idx+1]
            h, cos_i, cos_o, dot_wi_wo = self._compute_derived_features(wi, wo)
            h, cos_i, cos_o, dot_wi_wo = h[0], cos_i[0], cos_o[0], dot_wi_wo[0]

        result = {
            "pos": self.pos[idx],
            "normal": self.normal[idx],
            "uv": self.uv[idx],
            "wi_local": self.wi_local[idx],
            "wo_local": self.wo_local[idx],
            "h": h,
            "cos_i": cos_i,
            "cos_o": cos_o,
            "dot_wi_wo": dot_wi_wo,
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
        
        # Cache if enabled and we have space
        if (self.cache is not None and 
            len(self.cache) < self.cache_size):
            self.cache[idx] = result
        
        return result

    def _get_slice_items(self, idx: slice):
        """Get a slice of items efficiently."""
        # Get derived features
        if self.preload_features:
            h = self.h[idx]
            cos_i = self.cos_i[idx]
            cos_o = self.cos_o[idx]
            dot_wi_wo = self.dot_wi_wo[idx]
        else:
            wi = self.wi_local[idx]
            wo = self.wo_local[idx]
            h, cos_i, cos_o, dot_wi_wo = self._compute_derived_features(wi, wo)

        return {
            "pos": self.pos[idx],
            "normal": self.normal[idx],
            "uv": self.uv[idx],
            "wi_local": self.wi_local[idx],
            "wo_local": self.wo_local[idx],
            "h": h,
            "cos_i": cos_i,
            "cos_o": cos_o,
            "dot_wi_wo": dot_wi_wo,
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

    def _get_list_items(self, idx):
        """Get items from a list of indices."""
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        # Get derived features
        if self.preload_features:
            h = self.h[idx]
            cos_i = self.cos_i[idx]
            cos_o = self.cos_o[idx]
            dot_wi_wo = self.dot_wi_wo[idx]
        else:
            wi = self.wi_local[idx]
            wo = self.wo_local[idx]
            h, cos_i, cos_o, dot_wi_wo = self._compute_derived_features(wi, wo)

        return {
            "pos": self.pos[idx],
            "normal": self.normal[idx],
            "uv": self.uv[idx],
            "wi_local": self.wi_local[idx],
            "wo_local": self.wo_local[idx],
            "h": h,
            "cos_i": cos_i,
            "cos_o": cos_o,
            "dot_wi_wo": dot_wi_wo,
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

    def get_batch(self, indices: Union[slice, List[int]], batch_size: int = None):
        """Direct batch access method for maximum efficiency."""
        if batch_size is not None and isinstance(indices, slice):
            # Convert slice to list with batch_size limit
            start, stop, step = indices.indices(len(self))
            step = step or 1
            actual_indices = list(range(start, min(stop, start + batch_size), step))
            return self._get_list_items(actual_indices)
        else:
            return self.__getitem__(indices)

    def get_cache_stats(self):
        """Get cache performance statistics."""
        if self.cache is None:
            return "Cache disabled"
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_size
        }
import torch.utils.data as tud


class BatchSampler:
    """Custom batch sampler that can yield larger batches efficiently."""
    
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]
    
    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
import math
import warnings

warnings.filterwarnings(
    "ignore",
    message="Length of IterableDataset",
    category=UserWarning,
    module="torch.utils.data.dataloader"
)


class BatchedSubsetIterable(tud.IterableDataset):
    def __init__(self, subset: tud.Subset, batch_size: int):
        self.subset = subset          # torch.utils.data.Subset
        self.base = subset.dataset
        self.indices = subset.indices
        self.batch_size = batch_size
        self.length = len(self.indices)

    
    def _shard_indices(self):
        # Shard across data-loader workers
        info = torch.utils.data.get_worker_info()
        indices = self.indices
        if info is not None:
            per_worker = math.ceil(len(indices) / info.num_workers)
            start = info.id * per_worker
            end = min(start + per_worker, len(indices))
            indices = indices[start:end]

        # Optional: shard across distributed processes (Accelerate / DDP)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # Contiguous split
            per_proc = math.ceil(len(indices) / world)
            pstart = rank * per_proc
            pend = min(pstart + per_proc, len(indices))
            indices = indices[pstart:pend]
        return indices

    def __iter__(self):
        indices = self._shard_indices()
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield self.base[batch]
                batch = []
        if batch:
            yield self.base[batch]

    def __len__(self):
        # Length must match what a SINGLE worker+process will yield
        indices = self._shard_indices()
        n = len(indices)
        return (n + self.batch_size - 1) // self.batch_size


# ------------------------------------------------------------
# Optimized dataloader creation
# ------------------------------------------------------------
def create_dataloaders_optimized(path: str = "bsdf_samples.npz",
                                batch_size: int = 1024,
                                val_ratio: float = 0.1,
                                num_workers: int = 0,
                                shuffle_train: bool = True,
                                pin_memory: bool = True,
                                cache_size: int = 0,
                                preload_features: bool = True,
                                use_batch_sampler: bool = False,
                                **dataset_kwargs):
    """Create optimized dataloaders with various performance options."""
    
    full_ds = NeuralBRDFDatasetOptimized(
        npz_path=path, 
        cache_size=cache_size,
        preload_features=preload_features,
        **dataset_kwargs
    )
    
    val_size = int(len(full_ds) * val_ratio)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    if use_batch_sampler:

        
        training_set = BatchedSubsetIterable(train_ds, batch_size)
        validation_set = BatchedSubsetIterable(val_ds, batch_size)

        train_loader = DataLoader(
            training_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers)
        )
        val_loader = DataLoader(
            validation_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers)
        )
    else:
        # Standard DataLoader
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size,
            shuffle=shuffle_train, 
            num_workers=num_workers,
            pin_memory=pin_memory, 
            drop_last=False, 
            persistent_workers=bool(num_workers)
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory, 
            drop_last=False, 
            persistent_workers=bool(num_workers)
        )
    
    return train_loader, val_loader, full_ds


# ------------------------------------------------------------
# Performance testing utilities
# ------------------------------------------------------------
def benchmark_dataset(dataset, num_samples: int = 1000, batch_sizes: List[int] = None):
    """Benchmark dataset performance with different access patterns."""
    import time
    
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 256, 1024]
    
    print(f"Benchmarking dataset with {len(dataset)} samples")
    print(f"Testing {num_samples} samples with different batch sizes...")
    
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        if batch_size == 1:
            # Single item access
            for i in range(min(num_samples, len(dataset))):
                _ = dataset[i]
        else:
            # Batch access
            num_batches = min(num_samples // batch_size, len(dataset) // batch_size)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                _ = dataset[start_idx:end_idx]
        
        elapsed = time.time() - start_time
        samples_per_sec = num_samples / elapsed if batch_size == 1 else (num_batches * batch_size) / elapsed
        
        results[batch_size] = {
            'time': elapsed,
            'samples_per_sec': samples_per_sec
        }
        
        print(f"Batch size {batch_size:4d}: {elapsed:.3f}s, {samples_per_sec:.1f} samples/sec")
    
    return results



# ------------------------------------------------------------
# Minimal usage example
# ------------------------------------------------------------
if __name__ == "__main__":
    # Test optimized dataset
    print("Testing optimized dataset...")
    ds = NeuralBRDFDatasetOptimized(
        npz_path="bsdf_samples.npz",
        cache_size=1000,
        preload_features=True
    )
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Test single access
    print("\nTesting single access...")
    first = ds[0]
    print("Keys:", list(first.keys()))
    print("Sample shapes:", {k: v.shape for k, v in first.items()})
    
    # Test batch access
    print("\nTesting batch access...")
    batch = ds[0:100]
    print("Batch shapes:", {k: v.shape for k, v in batch.items()})
    
    # Test cache stats
    print(f"\nCache stats: {ds.get_cache_stats()}")
    
    # Run benchmark
    print("\nRunning performance benchmark...")
    benchmark_dataset(ds, num_samples=1000, batch_sizes=[1, 64, 256])