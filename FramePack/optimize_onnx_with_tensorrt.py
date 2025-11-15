#!/usr/bin/env python3
"""
Optimize ONNX models with TensorRT.

This script takes an ONNX model and builds an optimized TensorRT engine.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorrt as trt


# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: str,
    engine_path: str,
    *,
    fp16: bool = True,
    int8: bool = False,
    workspace_size_gb: int = 4,
    max_batch_size: int = 1,
    verbose: bool = False,
) -> bool:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model file
        engine_path: Path where the TensorRT engine will be saved
        fp16: Enable FP16 precision (default: True)
        int8: Enable INT8 precision (requires calibration, default: False)
        workspace_size_gb: Maximum workspace size in GB (default: 4)
        max_batch_size: Maximum batch size (default: 1)
        verbose: Enable verbose logging (default: False)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Building TensorRT Engine")
    print(f"{'='*80}\n")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output Engine: {engine_path}")
    print(f"FP16: {fp16}")
    print(f"INT8: {int8}")
    print(f"Workspace: {workspace_size_gb} GB")
    print(f"Max Batch Size: {max_batch_size}")
    print()

    # Check if ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        return False

    # Create output directory if needed
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            for error_idx in range(parser.num_errors):
                print(f"  Error {error_idx}: {parser.get_error(error_idx)}")
            return False

    print(f"Successfully parsed ONNX model")
    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")

    # Print input/output information
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  Input {i}: {input_tensor.name}, shape={input_tensor.shape}, dtype={input_tensor.dtype}")

    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  Output {i}: {output_tensor.name}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")

    # Create builder configuration
    config = builder.create_builder_config()

    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30))

    # Enable precision modes
    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("WARNING: FP16 not supported on this platform, using FP32")

    if int8:
        if builder.platform_has_fast_int8:
            print("Enabling INT8 precision")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here for best accuracy
            print("WARNING: INT8 enabled without calibration - accuracy may be reduced")
        else:
            print("WARNING: INT8 not supported on this platform")

    # Set optimization profile for dynamic shapes
    # This is important for models with dynamic input shapes
    profile = builder.create_optimization_profile()

    # Configure dynamic shapes for each input
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape

        # Check if any dimension is dynamic (-1)
        has_dynamic = any(dim == -1 for dim in input_shape)

        if has_dynamic:
            print(f"Setting up optimization profile for dynamic input: {input_tensor.name}")

            # Create min/opt/max shapes
            # Replace -1 with reasonable values
            min_shape = []
            opt_shape = []
            max_shape = []

            for dim in input_shape:
                if dim == -1:
                    # Batch dimension typically
                    min_shape.append(1)
                    opt_shape.append(max_batch_size)
                    max_shape.append(max_batch_size * 2)
                else:
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            print(f"  Min: {min_shape}")
            print(f"  Opt: {opt_shape}")
            print(f"  Max: {max_shape}")

    config.add_optimization_profile(profile)

    # Build engine
    print("\nBuilding TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return False

    # Save engine
    print(f"Saving TensorRT engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"\n{'='*80}")
    print(f"Successfully built TensorRT engine!")
    print(f"Engine saved to: {engine_path}")
    # Access the size via the nbytes attribute for IHostMemory objects
    engine_size_mb = len(bytes(serialized_engine)) / (1024*1024) if hasattr(serialized_engine, '__len__') else serialized_engine.nbytes / (1024*1024)
    print(f"Engine size: {engine_size_mb:.2f} MB")
    print(f"{'='*80}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ONNX models with TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with FP16
  python optimize_onnx_with_tensorrt.py \\
      --onnx-path Cache/onnx_models/flux_redux.onnx \\
      --output-path Cache/tensorrt_engines/flux_redux.engine

  # With custom workspace size
  python optimize_onnx_with_tensorrt.py \\
      --onnx-path Cache/onnx_models/flux_redux.onnx \\
      --output-path Cache/tensorrt_engines/flux_redux.engine \\
      --workspace-size 8 \\
      --fp16

  # Enable INT8 quantization (experimental)
  python optimize_onnx_with_tensorrt.py \\
      --onnx-path Cache/onnx_models/flux_redux.onnx \\
      --output-path Cache/tensorrt_engines/flux_redux.engine \\
      --int8
        """
    )

    parser.add_argument(
        '--onnx-path',
        type=str,
        required=True,
        help='Path to the input ONNX model file'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        help='Path for the output TensorRT engine file (default: same as ONNX with .engine extension)'
    )

    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Enable FP16 precision (default: True)'
    )

    parser.add_argument(
        '--no-fp16',
        action='store_false',
        dest='fp16',
        help='Disable FP16 precision (use FP32 instead)'
    )

    parser.add_argument(
        '--int8',
        action='store_true',
        default=False,
        help='Enable INT8 precision (experimental, may reduce accuracy)'
    )

    parser.add_argument(
        '--workspace-size',
        type=int,
        default=4,
        help='Maximum workspace size in GB (default: 4)'
    )

    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=1,
        help='Maximum batch size for dynamic shapes (default: 1)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set output path if not specified
    if args.output_path is None:
        onnx_path_obj = Path(args.onnx_path)
        args.output_path = str(onnx_path_obj.with_suffix('.engine'))

    # Build engine
    success = build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.output_path,
        fp16=args.fp16,
        int8=args.int8,
        workspace_size_gb=args.workspace_size,
        max_batch_size=args.max_batch_size,
        verbose=args.verbose,
    )

    if success:
        print("\n✓ TensorRT optimization completed successfully!")
        return 0
    else:
        print("\n✗ TensorRT optimization failed!")
        return 1


if __name__ == '__main__':
    exit(main())
