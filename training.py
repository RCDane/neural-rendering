import torch
import numpy as np
from torch.utils.data import dataloader
import dataset
from model import SimpleNeuralBSDF, TextureEncoder
from tqdm import tqdm
import argparse
import os
import json
try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

"""Training script for neural BSDF.

Additional CLI arguments:
    --save-dir PATH           Directory to store checkpoints (default: checkpoints)
    --save-interval N         Save checkpoint every N epochs (default: 10; 0 disables periodic saves)
The script always saves best_model.pth when validation improves and final_model.pth at the end.
"""

def training_step(neuralBSDF, batch):
    wi = batch['wi_local']
    wo = batch['wo_local']
    normal = batch['shading_normal']
    tangent = batch['tangent']
    roughness = batch['roughness']
    metallic = batch['metallic']
    albedo = batch['albedo']
    target_bsdf = batch['f']
    
    ndotl = torch.clamp(torch.sum(wi * normal, dim=-1, keepdim=True), min=0.0)
    ndotv = torch.clamp(torch.sum(wo * normal, dim=-1, keepdim=True), min=0.0)
    ndoth = torch.clamp(torch.sum((wi + wo) * 0.5 * normal, dim=-1, keepdim=True), min=0.0)
    ldoth = torch.clamp(torch.sum(wi * (wi + wo) * 0.5, dim=-1, keepdim=True), min=0.0)

    # Concatenate input features
    inputs = torch.cat([ndotl, ndotv, ndoth, ldoth, albedo, roughness, metallic], dim=-1)

    # Forward pass
    predicted_bsdf = neuralBSDF(inputs)
    predicted_bsdf = torch.exp(predicted_bsdf - 3.0)
    # Compute loss (Mean Squared Error)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(predicted_bsdf, target_bsdf)

    return loss

def epoch_step(neuralBSDF, dataloader, optimizer, accelerator, is_training = True):
    avg_loss = 0.0
    if is_training:
        neuralBSDF.train()
    else:
        neuralBSDF.eval()
    for b in tqdm(dataloader, desc="Training" if is_training else "Validation", unit="batch", total =len(dataloader), mininterval=2.0):
        loss = training_step(neuralBSDF, b)
        if is_training:
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
        avg_loss += loss.item()
    avg_loss /= max(1, len(dataloader))
    return avg_loss
    
         

def parse_args():
    parser = argparse.ArgumentParser(description="Train neural BSDF model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden layer dimension for BSDF model')
    parser.add_argument('--data', type=str, default='bsdf_samples.npz', help='Path to dataset npz')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N epochs (0 to disable)')
    parser.add_argument('--mixed-precision', type=str, default='no', choices=['no','fp16','bf16'], help='Mixed precision mode (accelerate)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log-every', type=int, default=1, help='Log every N epochs (main process)')
    # Removed legacy options: keep-last, no-save-best, no-final-save
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, training_loss, validation_loss, path):
    payload = {
        'epoch': epoch,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(payload, path)

    # Write sidecar JSON for quick inspection
    meta_path = path + '.json'
    meta = {k: payload[k] for k in ['epoch', 'training_loss', 'validation_loss']}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[checkpoint] Saved: {path}")


def main():
    # set cuda backend
    
    args = parse_args()
    training_set, validation_set, _ = dataset.create_dataloaders(
        path=args.data,
        batch_size=args.batch_size,
        num_workers=12,
        pin_memory=True,
    )
    print(len(training_set), "training samples")
    print(len(validation_set), "validation samples")



    if Accelerator is None:
        raise ImportError("accelerate not installed. Please run: pip install accelerate")
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.is_main_process:
        print(f"Accelerator initialized on device(s): {accelerator.device}, mixed_precision={accelerator.mixed_precision}")

    # optimize cuda
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Model parameters
    input_dim = 4 + 3   + 1 + 1  # wi, wo, albedo, roughness, metallic
    hidden_layer_dim = args.hidden_dim
    output_dim = 3  # RGB BSDF value
    
    neuralBSDF = SimpleNeuralBSDF(input_dim, hidden_layer_dim, output_dim)
    
    textureCompression = TextureEncoder(input_dim=3, hidden_layer_dim=64, output_dim=16)
    optimizer = torch.optim.AdamW(neuralBSDF.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Prepare dataloaders with accelerator (assuming training_set/validation_set implement iterable of dict tensors already on CPU; accelerator will move as needed)
    neuralBSDF, optimizer, training_set, validation_set = accelerator.prepare(neuralBSDF, optimizer, training_set, validation_set)

    ensure_dir(args.save_dir)
    best_val = float('inf')

    epoch_iter = range(args.epochs)
    if accelerator.is_main_process:
        epoch_iter = tqdm(epoch_iter)
    for epoch in epoch_iter:
        epoch_index = epoch + 1
        training_loss = epoch_step(neuralBSDF, training_set, optimizer, accelerator, is_training=True)
        validation_loss = epoch_step(neuralBSDF, validation_set, optimizer, accelerator, is_training=False)
        if accelerator.is_main_process and (epoch_index % args.log_every == 0):
            print(f"Epoch {epoch_index}/{args.epochs} - Training Loss: {training_loss:.6f}, Validation Loss: {validation_loss:.6f}")

        # Periodic checkpoint
        if accelerator.is_main_process and args.save_interval > 0 and (epoch_index % args.save_interval == 0):
            ckpt_name = f"ckpt_ep{epoch_index:04d}.pth"
            save_checkpoint(accelerator.unwrap_model(neuralBSDF), optimizer, epoch_index, training_loss, validation_loss, os.path.join(args.save_dir, ckpt_name))

        # Always save best model when improved
        if accelerator.is_main_process and validation_loss < best_val:
            best_val = validation_loss
            save_checkpoint(accelerator.unwrap_model(neuralBSDF), optimizer, epoch_index, training_loss, validation_loss, os.path.join(args.save_dir, 'best_model.pth'))

    # Always save final model
    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(neuralBSDF), optimizer, args.epochs, training_loss, validation_loss, os.path.join(args.save_dir, 'final_model.pth'))
        torch.save(accelerator.unwrap_model(neuralBSDF).state_dict(), os.path.join(args.save_dir, 'neural_bsdf_state_dict.pth'))
        print("Saved final model and state_dict.")
        
if __name__ == "__main__":
    main()