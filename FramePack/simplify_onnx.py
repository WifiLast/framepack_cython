#!/usr/bin/env python3
"""
Simplify ONNX models to make them more compatible with TensorRT.
"""

import argparse
from pathlib import Path

try:
    import onnx
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("WARNING: onnx-simplifier not installed. Install with: pip install onnx-simplifier")


def simplify_onnx_model(input_path: str, output_path: str = None, check_n: int = 3):
    """
    Simplify an ONNX model to make it more TensorRT-compatible.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to output simplified model (default: input_simplified.onnx)
        check_n: Number of times to check the simplified model
    """
    if not ONNXSIM_AVAILABLE:
        print("ERROR: onnx-simplifier is not installed")
        print("Install with: pip install onnx-simplifier")
        return False

    print(f"\n{'='*80}")
    print(f"Simplifying ONNX Model")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}")

    # Set output path
    if output_path is None:
        input_pathobj = Path(input_path)
        output_path = str(input_pathobj.parent / f"{input_pathobj.stem}_simplified{input_pathobj.suffix}")

    print(f"Output: {output_path}\n")

    # Load model
    print("Loading ONNX model...")
    try:
        model = onnx.load(input_path)
        print(f"✓ Loaded model: {len(model.SerializeToString()) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Simplify
    print("\nSimplifying model (this may take a few minutes)...")
    try:
        model_simplified, check = simplify(
            model,
            check_n=check_n,
            perform_optimization=True,
            skip_fuse_bn=False,
            skip_optimization=False,
        )

        if check:
            print("✓ Model simplified successfully")
            print(f"  Simplified size: {len(model_simplified.SerializeToString()) / (1024*1024):.2f} MB")

            # Save
            print(f"\nSaving simplified model to: {output_path}")
            onnx.save(model_simplified, output_path)
            print("✓ Saved successfully")

            print(f"\n{'='*80}")
            print("Simplification complete!")
            print(f"{'='*80}\n")
            return True
        else:
            print("⚠ Simplification may have introduced errors")
            print("  Saving anyway, but verify the model works correctly")
            onnx.save(model_simplified, output_path)
            return True

    except Exception as e:
        print(f"✗ Simplification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simplify ONNX models for better TensorRT compatibility"
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input ONNX model'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        help='Path to output simplified model (default: input_simplified.onnx)'
    )
    parser.add_argument(
        '--check-n',
        type=int,
        default=3,
        help='Number of times to check the simplified model (default: 3)'
    )

    args = parser.parse_args()

    success = simplify_onnx_model(
        input_path=args.input_path,
        output_path=args.output_path,
        check_n=args.check_n,
    )

    if success:
        print("\n✓ Model simplification completed successfully!")
        return 0
    else:
        print("\n✗ Model simplification failed!")
        return 1


if __name__ == '__main__':
    exit(main())
