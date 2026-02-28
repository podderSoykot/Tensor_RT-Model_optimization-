# TensorRT API Fix

## Problem
The error `AttributeError: 'tensorrt_bindings.tensorrt.IBuilderConfig' object has no attribute 'max_workspace_size'` occurs because TensorRT 8.5+ changed the API.

## Solution

### Option 1: Use Step 6 (Recommended)
Skip Step 5 and use Step 6 which uses Ultralytics' built-in TensorRT export. This is simpler and more reliable.

### Option 2: Fix Step 5 Code
Replace the line in Step 5:
```python
config.max_workspace_size = 1 << 30  # 1GB
```

With:
```python
# Fixed for TensorRT 8.5+
try:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
except (AttributeError, TypeError):
    # Fallback for older versions
    try:
        config.max_workspace_size = 1 << 30
    except:
        pass
```

### Option 3: Use the Fixed Function
A fixed version is available in the notebook (after Step 5). You can use that function instead.

## Quick Fix in Colab
In the Step 5 cell, find this line:
```python
config.max_workspace_size = 1 << 30  # 1GB
```

Replace it with:
```python
try:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
except (AttributeError, TypeError):
    try:
        config.max_workspace_size = 1 << 30
    except:
        pass
```

