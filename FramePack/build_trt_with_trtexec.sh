#!/bin/bash
# Build TensorRT engine using trtexec (TensorRT's official tool)
# This is more robust than the Python API for debugging

ONNX_PATH=${1:-"Cache/onnx_models/flux_redux_optimized.onnx"}
ENGINE_PATH=${2:-"Cache/tensorrt_engines/flux_redux.engine"}
WORKSPACE_MB=${3:-4096}

echo "=============================================================================="
echo "Building TensorRT Engine with trtexec"
echo "=============================================================================="
echo "Input ONNX: $ONNX_PATH"
echo "Output Engine: $ENGINE_PATH"
echo "Workspace: ${WORKSPACE_MB} MB"
echo ""

# Create output directory
mkdir -p "$(dirname "$ENGINE_PATH")"

# Run trtexec with verbose output
trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --workspace=$WORKSPACE_MB \
    --verbose \
    --dumpLayerInfo \
    --exportLayerInfo=trt_layer_info.json \
    2>&1 | tee trtexec_build.log

# Check result
if [ -f "$ENGINE_PATH" ]; then
    echo ""
    echo "=============================================================================="
    echo "✓ TensorRT engine built successfully!"
    echo "Engine: $ENGINE_PATH"
    echo "Size: $(du -h "$ENGINE_PATH" | cut -f1)"
    echo "=============================================================================="
    exit 0
else
    echo ""
    echo "=============================================================================="
    echo "✗ Failed to build TensorRT engine"
    echo "Check trtexec_build.log for details"
    echo "=============================================================================="
    exit 1
fi
