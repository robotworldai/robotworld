import torch
import argparse
import os

def main():
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Export TorchScript model to ONNX')
    parser.add_argument('--jit-path', type=str, default="checkpoint/policy.pt",
                       help='Path to the TorchScript model file (.pt)')
    parser.add_argument('--onnx-path', type=str, default="exported/policy.onnx",
                       help='Path for the output ONNX file (default: same directory as jit_path with .onnx extension)')
    parser.add_argument('--input-shape', type=int, nargs='+', default=[1, 480],
                       help='Input shape for the model (default: [1, 480])')
    
    args = parser.parse_args()
    
    # Check if the input TorchScript file exists
    if not os.path.exists(args.jit_path):
        raise FileNotFoundError(f"TorchScript file not found: {args.jit_path}")

    onnx_path = args.onnx_path    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Load TorchScript model
    ts = torch.jit.load(args.jit_path).eval()
    
    # Create dummy input
    if len(args.input_shape) == 1:
        # If only one dimension is provided, assume batch size of 1
        input_shape = [1] + args.input_shape
    else:
        input_shape = args.input_shape
    
    dummy = torch.randn(*input_shape)
    
    # Export to ONNX
    with torch.inference_mode():
        torch.onnx.export(
            ts,
            dummy,
            onnx_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=17,
            do_constant_folding=True,
        )
    
    print(f"Exported: {onnx_path}")

if __name__ == "__main__":
    main()