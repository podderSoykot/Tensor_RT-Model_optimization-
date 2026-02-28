# Fixed TensorRT build_engine function for Colab
# Use this if you need manual TensorRT conversion

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_path, engine_path, precision='fp16'):
    """
    Convert ONNX model to TensorRT engine (Fixed for TensorRT 8.5+)
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set memory pool limit (Fixed for TensorRT 8.5+)
    # TensorRT 8.5+ uses set_memory_pool_limit instead of max_workspace_size
    try:
        # Try new API first (TensorRT 8.5+)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        print("Using TensorRT 8.5+ API")
    except (AttributeError, TypeError):
        # Fallback for older TensorRT versions
        try:
            config.max_workspace_size = 1 << 30  # 1GB
            print("Using older TensorRT API")
        except AttributeError:
            print("Warning: Could not set workspace size - using default")
    
    # Set precision
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")
    else:
        print("Using FP32 precision")
    
    # Build engine
    print("Building TensorRT engine... This may take a few minutes.")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to build engine")
        return None
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to: {engine_path}")
    return engine

