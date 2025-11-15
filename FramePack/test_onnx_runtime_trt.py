#!/usr/bin/env python3
"""
Test ONNX model with ONNX Runtime + TensorRT Execution Provider.

This is often more reliable than building standalone TensorRT engines,
especially for complex models with operations TensorRT doesn't fully support.
"""

import argparse
import numpy as np
import onnxruntime as ort
import time


def test_onnx_with_tensorrt_ep(
    onnx_path: str,
    use_fp16: bool = True,
    trt_cache_dir: str = "./trt_cache",
    use_tensorrt: bool = True,
):
    """
    Test ONNX model using ONNX Runtime with TensorRT Execution Provider.

    This approach is more flexible than standalone TensorRT engines because:
    - Automatically falls back to CUDA/CPU for unsupported ops
    - Better error handling
    - Easier to debug
    """

    print(f"\n{'='*80}")
    print(f"Testing ONNX Runtime with TensorRT Execution Provider")
    print(f"{'='*80}\n")
    print(f"ONNX Model: {onnx_path}")
    print(f"FP16: {use_fp16}")
    print(f"TensorRT Cache: {trt_cache_dir}\n")

    # Configure TensorRT Execution Provider
    trt_provider_options = {
        'trt_fp16_enable': use_fp16,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': trt_cache_dir,
        'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
    }

    # Create session with providers in priority order
    # TensorRT -> CUDA -> CPU
    if use_tensorrt:
        providers = [
            ('TensorrtExecutionProvider', trt_provider_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    else:
        print("  Using CUDA execution provider (TensorRT disabled)")
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    print("Creating ONNX Runtime session...")
    print(f"  Providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")

    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print("✓ Session created successfully")
    except Exception as e:
        print(f"✗ Failed to create session: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get actual providers used
    actual_providers = session.get_providers()
    print(f"\n  Active providers: {actual_providers}")

    if 'TensorrtExecutionProvider' in actual_providers:
        print("  ✓ TensorRT Execution Provider is active!")
    elif 'CUDAExecutionProvider' in actual_providers:
        print("  ⚠ Using CUDA (TensorRT not available)")
    else:
        print("  ⚠ Using CPU (no GPU acceleration)")

    # Get model inputs
    print("\nModel Inputs:")
    inputs = session.get_inputs()
    for inp in inputs:
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    # Get model outputs
    print("\nModel Outputs:")
    outputs = session.get_outputs()
    for out in outputs:
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Create dummy input
    print("\nCreating dummy input...")
    input_dict = {}
    for inp in inputs:
        shape = inp.shape
        # Replace dynamic dimensions with concrete values
        concrete_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim < 0:
                concrete_shape.append(1)  # Use batch=1
            else:
                concrete_shape.append(dim)

        # Create random input
        if 'float' in inp.type:
            input_data = np.random.randn(*concrete_shape).astype(np.float32)
        elif 'int64' in inp.type:
            input_data = np.random.randint(0, 1000, size=concrete_shape, dtype=np.int64)
        else:
            input_data = np.random.randn(*concrete_shape).astype(np.float32)

        input_dict[inp.name] = input_data
        print(f"  {inp.name}: {input_data.shape} ({input_data.dtype})")

    # Run inference (warm-up)
    print("\nWarm-up inference...")
    try:
        _ = session.run(None, input_dict)
        print("✓ Warm-up successful")
    except Exception as e:
        print(f"✗ Warm-up failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Benchmark
    print("\nBenchmarking (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        result = session.run(None, input_dict)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        print(f"  Iteration {i+1}: {times[-1]:.2f} ms")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nBenchmark Results:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min(times):.2f} ms")
    print(f"  Max: {max(times):.2f} ms")

    # Check output shapes
    print(f"\nOutput Results:")
    for i, out in enumerate(outputs):
        print(f"  {out.name}: {result[i].shape} ({result[i].dtype})")

    print(f"\n{'='*80}")
    print("✓ ONNX Runtime + TensorRT test completed successfully!")
    print(f"{'='*80}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test ONNX model with ONNX Runtime + TensorRT EP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This approach is recommended over standalone TensorRT engines because:
- More robust error handling
- Automatic fallback for unsupported operations
- Easier debugging
- Still gets TensorRT acceleration where possible

Example:
  python test_onnx_runtime_trt.py Cache/onnx_models/flux_redux_optimized.onnx
        """
    )

    parser.add_argument(
        'onnx_path',
        type=str,
        help='Path to ONNX model file'
    )

    parser.add_argument(
        '--no-fp16',
        action='store_false',
        dest='fp16',
        help='Disable FP16 (use FP32)'
    )

    parser.add_argument(
        '--trt-cache-dir',
        type=str,
        default='./trt_cache',
        help='Directory for TensorRT engine cache (default: ./trt_cache)'
    )

    parser.add_argument(
        '--no-tensorrt',
        action='store_false',
        dest='tensorrt',
        help='Disable TensorRT (use CUDA only)'
    )

    args = parser.parse_args()

    success = test_onnx_with_tensorrt_ep(
        onnx_path=args.onnx_path,
        use_fp16=args.fp16,
        trt_cache_dir=args.trt_cache_dir,
        use_tensorrt=args.tensorrt,
    )

    if success:
        print("\n✓ Test completed successfully!")
        print("\nNext steps:")
        print("  1. TensorRT engines are cached in:", args.trt_cache_dir)
        print("  2. Use ONNX Runtime in your application for inference")
        print("  3. First run will be slow (engine building), subsequent runs will be fast")
        return 0
    else:
        print("\n✗ Test failed!")
        return 1


if __name__ == '__main__':
    exit(main())
