#!/usr/bin/env python3
"""
Optimize ONNX models using onnxoptimizer for better TensorRT compatibility.
"""

import argparse
from pathlib import Path

try:
    import onnx
    from onnxoptimizer import optimize
    ONNXOPTIMIZER_AVAILABLE = True
except ImportError:
    ONNXOPTIMIZER_AVAILABLE = False
    print("WARNING: onnxoptimizer not installed. Install with: pip install onnxoptimizer")


def optimize_onnx_model(
    input_path: str,
    output_path: str = None,
    optimization_level: str = 'all',
    fixed_point: bool = True,
):
    """
    Optimize an ONNX model using onnxoptimizer.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to output optimized model (default: input_optimized.onnx)
        optimization_level: Optimization level ('basic', 'extended', 'all')
        fixed_point: Apply optimizations until fixed point
    """
    if not ONNXOPTIMIZER_AVAILABLE:
        print("ERROR: onnxoptimizer is not installed")
        print("Install with: pip install onnxoptimizer")
        return False

    print(f"\n{'='*80}")
    print(f"Optimizing ONNX Model with onnxoptimizer")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}")

    # Set output path
    if output_path is None:
        input_pathobj = Path(input_path)
        output_path = str(input_pathobj.parent / f"{input_pathobj.stem}_optimized{input_pathobj.suffix}")

    print(f"Output: {output_path}")
    print(f"Optimization Level: {optimization_level}")
    print(f"Fixed Point: {fixed_point}\n")

    # Load model
    print("Loading ONNX model...")
    try:
        model = onnx.load(input_path)
        original_size = len(model.SerializeToString()) / (1024*1024)
        print(f"✓ Loaded model: {original_size:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Get optimization passes based on level
    if optimization_level == 'basic':
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm',
        ]
    elif optimization_level == 'extended':
        passes = [
            'eliminate_deadend',
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_monotone_argmax',
            'eliminate_nop_pad',
            'eliminate_nop_transpose',
            'eliminate_unused_initializer',
            'extract_constant_to_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_log_softmax',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm',
        ]
    else:  # 'all'
        # Use all available optimization passes
        passes = None  # None means use all passes

    # Optimize
    print("\nOptimizing model...")
    print("  This may take a few minutes for large models...")
    try:
        if passes is None:
            # Use all available optimizations
            optimized_model = optimize(model, fixed_point=fixed_point)
        else:
            # Use specific passes
            optimized_model = optimize(model, passes=passes, fixed_point=fixed_point)

        optimized_size = len(optimized_model.SerializeToString()) / (1024*1024)
        print(f"✓ Model optimized successfully")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Optimized size: {optimized_size:.2f} MB")
        print(f"  Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")

        # Validate optimized model
        print("\nValidating optimized model...")
        try:
            onnx.checker.check_model(optimized_model)
            print("✓ Optimized model is valid")
        except Exception as e:
            print(f"⚠ Validation warning: {e}")
            print("  Model may still work, but verify functionality")

        # Save
        print(f"\nSaving optimized model to: {output_path}")
        onnx.save(optimized_model, output_path)
        print("✓ Saved successfully")

        print(f"\n{'='*80}")
        print("Optimization complete!")
        print(f"{'='*80}\n")
        return True

    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ONNX models using onnxoptimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  python optimize_onnx.py Cache/onnx_models/flux_redux.onnx

  # Extended optimization
  python optimize_onnx.py Cache/onnx_models/flux_redux.onnx \\
      --level extended

  # All optimizations with custom output path
  python optimize_onnx.py Cache/onnx_models/flux_redux.onnx \\
      --output-path Cache/onnx_models/flux_redux_opt.onnx \\
      --level all
        """
    )

    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input ONNX model'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        help='Path to output optimized model (default: input_optimized.onnx)'
    )

    parser.add_argument(
        '--level',
        type=str,
        choices=['basic', 'extended', 'all'],
        default='all',
        help='Optimization level (default: all)'
    )

    parser.add_argument(
        '--no-fixed-point',
        action='store_false',
        dest='fixed_point',
        help='Disable fixed-point optimization (single pass only)'
    )

    args = parser.parse_args()

    success = optimize_onnx_model(
        input_path=args.input_path,
        output_path=args.output_path,
        optimization_level=args.level,
        fixed_point=args.fixed_point,
    )

    if success:
        print("\n✓ Model optimization completed successfully!")
        print("\nNext step: Convert to TensorRT engine:")
        output = args.output_path if args.output_path else str(Path(args.input_path).parent / f"{Path(args.input_path).stem}_optimized.onnx")
        print(f"  python optimize_onnx_with_tensorrt.py --onnx-path {output}")
        return 0
    else:
        print("\n✗ Model optimization failed!")
        return 1


if __name__ == '__main__':
    exit(main())
