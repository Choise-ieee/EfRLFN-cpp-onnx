import torch
import argparse
import os
from code.model import EfRLFN

def export_onnx(args):
    # 1. Initialize the network
    model = EfRLFN(upscale=args.scale)
    
    # 2. Load weights (simplified logic from your script)
    if args.weights.endswith('.pt'):
        state_dict = torch.load(args.weights)
    elif args.weights.endswith('.ckpt'):
        checkpoint = torch.load(args.weights)
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('network.'):
                state_dict[k[len("network."):]] = v
    else:
        raise ValueError("Unsupported weight format")
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 3. Create dummy input
    # Input size usually doesn't matter for fully convolutional networks, 
    # but standard is often 256x256 or similar.
    dummy_input = torch.randn(1, 3, 64, 64)

    # 4. Export
    onnx_path = args.output if args.output else "efrlfn_x{}.onnx".format(args.scale)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        verbose=False, 
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={ # Allow dynamic input sizes
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"Model exported to {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, required=True)
    parser.add_argument('-s', '--scale', type=int, required=True)
    parser.add_argument('-o', '--output', type=str, default="")
    args = parser.parse_args()
    export_onnx(args)
