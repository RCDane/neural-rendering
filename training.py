import torch
import numpy as np
from torch.utils.data import dataloader
import dataset
from model import SimpleNeuralBSDF, TextureEncoder, save_checkpoint, load_checkpoint, parse_model_input_values
from tqdm import tqdm
import argparse
import os
import json

from accelerate import Accelerator




"""Training script for neural BSDF.

Additional CLI arguments:
    --save-dir PATH           Directory to store checkpoints (default: checkpoints)
    --save-interval N         Save checkpoint every N epochs (default: 10; 0 disables periodic saves)
The script always saves best_model.pth when validation improves and final_model.pth at the end.
"""



def training_step(neuralBSDF, batch):
    # wi = batch['wi_local']
    # wo = batch['wo_local']
    # normal = batch['shading_normal']
    # tangent = batch['tangent']
    # roughness = batch['roughness']
    # metallic = batch['metallic']
    # albedo = batch['albedo']
    target_bsdf = batch['f'].squeeze(0)
    target_color = target_bsdf*batch['albedo'].unsqueeze(0)
    
    # ndotl = torch.clamp(torch.sum(wi * normal, dim=-1, keepdim=True), min=0.0)
    # ndotv = torch.clamp(torch.sum(wo * normal, dim=-1, keepdim=True), min=0.0)
    # ndoth = torch.clamp(torch.sum((wi + wo) * 0.5 * normal, dim=-1, keepdim=True), min=0.0)
    # ldoth = torch.clamp(torch.sum(wi * (wi + wo) * 0.5, dim=-1, keepdim=True), min=0.0)

    # Concatenate input features
    # inputs = torch.cat([ndotl, ndotv, ndoth, ldoth, albedo, roughness, metallic], dim=-1)
    inputs = parse_model_input_values(
        input_sizes=[('ndotl', 1), ('ndotv', 1), ('ndoth', 1), ('ldoth', 1), ('albedo', 3), ('roughness', 1), ('metallic', 1)],
        input_values=batch,
        batch_size=target_bsdf.shape[0]
    )
    # Forward pass
    predicted_bsdf = neuralBSDF(inputs)
    predicted_bsdf = torch.exp(predicted_bsdf - 3.0)
    # Compute loss (Mean Squared Error)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(predicted_bsdf, target_color)

    return loss

def epoch_step(neuralBSDF, dataloader, optimizer, accelerator, is_training = True):
    avg_loss = 0.0
    if is_training:
        neuralBSDF.train()
    else:
        neuralBSDF.eval()
    for b in tqdm(dataloader, desc="Training" if is_training else "Validation", unit="batch", total=len(dataloader)):
        loss = training_step(neuralBSDF, b)
        if is_training:
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
        avg_loss += loss.item()
    avg_loss /= max(1, len(dataloader))
    return avg_loss
    
default_shape = """
input_shape:
  - ndotl: 1
  - ndotv: 1
  - ndoth: 1
  - ldoth: 1
  - albedo: 3
  - roughness: 1
  - metallic: 1
"""

def parse_input_shape(input_shape_str: str):
    import yaml
    if not input_shape_str:
        return None
    try:
        obj = yaml.safe_load(input_shape_str)
        # Allow passing either the list directly or a dict under 'input_shape'
        if isinstance(obj, dict) and 'input_shape' in obj and isinstance(obj['input_shape'], list):
            obj = obj['input_shape']
        if isinstance(obj, list):
            result = []
            seen = set()
            for entry in obj:
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError(f"Invalid entry in input_shape list: {entry}")
                k, v = next(iter(entry.items()))
                if k in seen:
                    raise ValueError(f"Duplicate key in input_shape: {k}")
                if not isinstance(k, str) or not isinstance(v, int):
                    raise ValueError(f"Input shape entries must be (str -> int): {entry}")
                result.append((k, int(v)))
                seen.add(k)
            return result
        if isinstance(obj, dict):
            # Fallback: mapping form
            return [(k, int(v)) for k, v in obj.items()]
        raise ValueError("input_shape must be a YAML list of single-key dicts or a mapping.")
    except Exception as e:
        print(f"Error parsing input shape YAML: {e}")
        return None

def parse_args():
    import yaml
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", help="Path to YAML config file", metavar="FILE")
    args, remaining_argv = config_parser.parse_known_args()
    defaults = {}
    if args.config:
        with open(args.config, 'r') as f:
            defaults = yaml.safe_load(f)
        if 'input_shape' in defaults:
            # convert to yaml string
            defaults['input_shape'] = yaml.dump(defaults['input_shape'])
    
    
    parser = argparse.ArgumentParser(
        description="My script description",
        # Inherit from the config_parser to get the --config argument
        parents=[config_parser] 
    )
    
    parser = argparse.ArgumentParser(description="Train neural BSDF model")
    parser.add_argument('-c', '--config', type=str, help='Path to config file (JSON)', default='')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden layer dimension for BSDF model')
    parser.add_argument('--data', type=str, default='bsdf_samples.npz', help='Path to dataset npz')
    parser.add_argument('--input-shape', type=str, default=default_shape, help='JSON string defining input shape (overrides default)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N epochs (0 to disable)')
    parser.add_argument('--mixed-precision', type=str, default='no', choices=['no','fp16','bf16'], help='Mixed precision mode (accelerate)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log-every', type=int, default=1, help='Log every N epochs (main process)')
    parser.set_defaults(**defaults)
    final_args = parser.parse_args(remaining_argv)
    # print arguments:
    print("Final arguments:")
    for arg in vars(final_args):
        print(f"  {arg}: {getattr(final_args, arg)}")
    # Removed legacy options: keep-last, no-save-best, no-final-save
    return final_args


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    logger.info(f"Using GPU(s): {gpus_to_use}")

assign_free_gpus(2000, max_gpus=1)

def main():
    # set cuda backend
    import dataset_optimized
    args = parse_args()
    
    

    training_set, validation_set, _ = dataset_optimized.create_dataloaders_optimized(
        path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        use_batch_sampler=True,
    )
    print(len(training_set), "training samples")
    print(len(validation_set), "validation samples")


    


    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.is_main_process:
        print(f"Accelerator initialized on device(s): {accelerator.device}, mixed_precision={accelerator.mixed_precision}")

    # optimize cuda
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    input_shape= parse_input_shape(args.input_shape)
    
    
    print("input shape:", input_shape)
    # Model parameters
    input_dim = sum([s[1] for s in input_shape])  # wi, wo, albedo, roughness, metallic
    hidden_layer_dim = args.hidden_dim
    output_dim = 3  # RGB BSDF value
    
    neuralBSDF = SimpleNeuralBSDF(input_dim, hidden_layer_dim, output_dim)
    
    textureCompression = TextureEncoder(input_dim=3, hidden_layer_dim=64, output_dim=16)
    optimizer = torch.optim.AdamW(neuralBSDF.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Prepare dataloaders with accelerator (assuming training_set/validation_set implement iterable of dict tensors already on CPU; accelerator will move as needed)
    
    neuralBSDF, optimizer, training_set, validation_set,  = accelerator.prepare(neuralBSDF, optimizer, training_set, validation_set)

    
    #learning rate


    ensure_dir(args.save_dir)
    best_val = float('inf')

    epoch_iter = range(args.epochs)
    if accelerator.is_main_process:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        epoch_index = epoch + 1
        training_loss = epoch_step(neuralBSDF, training_set, optimizer, accelerator, is_training=True)
        validation_loss = epoch_step(neuralBSDF, validation_set, optimizer, accelerator, is_training=False)

        if accelerator.is_main_process and (epoch_index % args.log_every == 0):
            print(f"Epoch {epoch_index}/{args.epochs} - Training Loss: {training_loss:.6f}, Validation Loss: {validation_loss:.6f}")

        # Periodic checkpoint
        if accelerator.is_main_process and args.save_interval > 0 and (epoch_index % args.save_interval == 0):
            ckpt_name = f"ckpt_ep{epoch_index:04d}.pth"
            save_checkpoint(accelerator.unwrap_model(neuralBSDF), optimizer, epoch_index, training_loss, validation_loss, input_shape, os.path.join(args.save_dir, ckpt_name))

        # Always save best model when improved
        if accelerator.is_main_process and validation_loss < best_val:
            best_val = validation_loss
            save_checkpoint(accelerator.unwrap_model(neuralBSDF), optimizer, epoch_index, training_loss, validation_loss, input_shape, os.path.join(args.save_dir, 'best_model.pth'))

    # Always save final model
    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(neuralBSDF), optimizer, args.epochs, training_loss, validation_loss, input_shape, os.path.join(args.save_dir, 'final_model.pth'))
        torch.save(accelerator.unwrap_model(neuralBSDF).state_dict(), os.path.join(args.save_dir, 'neural_bsdf_state_dict.pth'))
        print("Saved final model and state_dict.")
        
if __name__ == "__main__":
    main()