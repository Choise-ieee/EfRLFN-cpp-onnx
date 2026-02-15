# EfRLFN-cpp-onnx
The cpp and onnx version of papar "Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content" 
(https://arxiv.org/abs/2602.11339)
(https://github.com/EvgeneyBogatyrev/EfRLFN)

# Export Model to ONNX (Python)-export_onnx.py in files
```python
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

```

We can the command: python export_onnx.py -w weigths\EfRLFN-4x-model.pt -s 4 or python export_onnx.py -w weigths\EfRLFN-2x-model.pt -s 2
Then the onnx files can be found:
<img width="1164" height="460" alt="image" src="https://github.com/user-attachments/assets/3a62b148-c95a-4df9-a903-a654dc42c332" />

# windows system inference
According to the main.cpp in files,select efrlfn_x4.onnx or efrlfn_x2.onnx

***I sucess achieve in VS2019, cuda 12.1 and onnxrunningtime-1.18 in windows10 and windows server2012. 1.C++ inference in step 11, If you use cudnn9.x please use onnxruunming-time 18.1.1, and if you use cudnn8.x use onnx 18.1.0***
<img width="3106" height="1546" alt="70beabed3618b8d205b76ddc96a7e10b" src="https://github.com/user-attachments/assets/35615fd6-6252-4b5b-8525-992fe2560f57" />

# Inference results
The orignal Picture(1K)
<img width="2314" height="1438" alt="image" src="https://github.com/user-attachments/assets/f82d97d9-1fc5-4d77-bfd8-9bffa2ff910d" />


The 2X scale+ picture(2K output)
<img width="2314" height="1438" alt="image" src="https://github.com/user-attachments/assets/87f7bcfa-3dcc-4c85-b8fb-0e972f57fe83" />


The 4X scale+ picture(4K output)
<img width="2314" height="1438" alt="image" src="https://github.com/user-attachments/assets/6c7915dd-031c-4643-ba97-b8505e612171" />


# PS.
We also can enable GPU calculation by enable the thress lines codings
<img width="1796" height="1032" alt="image" src="https://github.com/user-attachments/assets/d75708fa-6933-493f-956e-975a5be0ce6d" />

Also Thanks for their works
	arXiv:2602.11339 [cs.CV]
 	(or arXiv:2602.11339v1 [cs.CV] for this version)
 
https://doi.org/10.48550/arXiv.2602.11339

