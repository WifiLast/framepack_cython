#!/usr/bin/env python3
"""
Check ONNX model compatibility and structure.
"""

import argparse
import onnx
from onnx import checker, shape_inference
import onnxruntime as ort


def check_onnx_model(onnx_path: str, run_inference_test: bool = False):
    """Check ONNX model validity and compatibility."""

    print(f"\n{'='*80}")
    print(f"Checking ONNX Model: {onnx_path}")
    print(f"{'='*80}\n")

    # Load the model
    print("Loading ONNX model...")
    try:
        model = onnx.load(onnx_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Check model validity
    print("\nValidating model structure...")
    try:
        checker.check_model(model)
        print("✓ Model is valid")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False

    # Get model info
    print("\nModel Information:")
    print(f"  IR Version: {model.ir_version}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    print(f"  Opset Version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
    print(f"  Graph Name: {model.graph.name}")

    # Check inputs
    print(f"\nInputs ({len(model.graph.input)}):")
    for idx, input_tensor in enumerate(model.graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]
        dtype = input_tensor.type.tensor_type.elem_type
        print(f"  {idx}. {input_tensor.name}")
        print(f"     Shape: {shape}")
        print(f"     Type: {onnx.TensorProto.DataType.Name(dtype)}")

    # Check outputs
    print(f"\nOutputs ({len(model.graph.output)}):")
    for idx, output_tensor in enumerate(model.graph.output):
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in output_tensor.type.tensor_type.shape.dim]
        dtype = output_tensor.type.tensor_type.elem_type
        print(f"  {idx}. {output_tensor.name}")
        print(f"     Shape: {shape}")
        print(f"     Type: {onnx.TensorProto.DataType.Name(dtype)}")

    # Check for problematic ops
    print(f"\nOperators in model ({len(model.graph.node)} total):")
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")

    # Known problematic ops for TensorRT
    problematic_ops = ['Loop', 'If', 'Scan', 'NonMaxSuppression', 'NonZero', 'Unique']
    found_problematic = [op for op in problematic_ops if op in op_types]

    if found_problematic:
        print(f"\n⚠ WARNING: Found potentially problematic operators for TensorRT:")
        for op in found_problematic:
            print(f"  - {op} ({op_types[op]} occurrences)")
        print("  These operators may cause TensorRT build failures or require workarounds")

    # Try shape inference
    print("\nPerforming shape inference...")
    try:
        inferred_model = shape_inference.infer_shapes(model)
        print("✓ Shape inference successful")
    except Exception as e:
        print(f"⚠ Shape inference failed: {e}")

    # Test with ONNX Runtime
    if run_inference_test:
        print("\nTesting with ONNX Runtime...")
        try:
            session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            print("✓ ONNX Runtime can load the model")

            # Get provider
            provider = session.get_providers()[0]
            print(f"  Using provider: {provider}")

            # Show input/output info from runtime
            print("\n  Runtime Input Details:")
            for input_meta in session.get_inputs():
                print(f"    {input_meta.name}: {input_meta.shape} ({input_meta.type})")

            print("\n  Runtime Output Details:")
            for output_meta in session.get_outputs():
                print(f"    {output_meta.name}: {output_meta.shape} ({output_meta.type})")

        except Exception as e:
            print(f"✗ ONNX Runtime test failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Model check complete!")
    print(f"{'='*80}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Check ONNX model compatibility")
    parser.add_argument('onnx_path', type=str, help='Path to ONNX model file')
    parser.add_argument('--test-inference', action='store_true', help='Test with ONNX Runtime')
    args = parser.parse_args()

    check_onnx_model(args.onnx_path, run_inference_test=args.test_inference)


if __name__ == '__main__':
    main()
