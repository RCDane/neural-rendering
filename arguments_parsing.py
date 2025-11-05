import argparse
import yaml

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