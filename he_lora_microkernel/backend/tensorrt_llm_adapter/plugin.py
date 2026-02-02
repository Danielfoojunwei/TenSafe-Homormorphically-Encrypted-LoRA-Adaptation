"""
TensorRT Plugin for HE-LoRA Delta Injection

This module defines a TensorRT plugin that adds HE-LoRA deltas
to attention projection outputs.

The plugin:
  1. Receives projection output from the model
  2. Reads delta from shared memory buffer
  3. Adds delta to output
  4. Returns modified output

This enables delta injection within the TensorRT execution graph.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import ctypes

# Plugin configuration
PLUGIN_NAME = "HELoRAProjection"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "helora"


@dataclass
class PluginConfig:
    """Configuration for HE-LoRA projection plugin."""
    layer_idx: int
    projection_type: str  # "q", "k", "v", "o"
    hidden_size: int
    batch_size: int

    # Shared memory configuration
    shm_name: str = ""
    shm_offset: int = 0

    # Delta tensor layout
    delta_dtype: str = "float16"


class HELoRAProjectionPlugin:
    """
    TensorRT plugin for HE-LoRA delta injection.

    This is a Python wrapper around the TensorRT plugin. The actual
    plugin implementation would be in C++/CUDA for production use.

    For development/testing, this class provides the plugin interface
    that can be used with TensorRT-LLM's Python bindings.
    """

    def __init__(self, config: PluginConfig):
        """
        Initialize plugin.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self._delta_buffer: Optional[np.ndarray] = None
        self._shm_handle = None

    @property
    def name(self) -> str:
        return f"{PLUGIN_NAME}_{self.config.layer_idx}_{self.config.projection_type}"

    def set_delta_buffer(self, buffer: np.ndarray) -> None:
        """
        Set the delta buffer to add to projection output.

        Args:
            buffer: Delta tensor (batch_size, seq_len, hidden_size)
        """
        if buffer.dtype != np.float16:
            buffer = buffer.astype(np.float16)
        self._delta_buffer = buffer

    def attach_shared_memory(self, shm_name: str, shm_offset: int = 0) -> None:
        """
        Attach to shared memory region for delta data.

        Args:
            shm_name: Shared memory region name
            shm_offset: Offset within region for this plugin's delta
        """
        import mmap
        import os

        self.config.shm_name = shm_name
        self.config.shm_offset = shm_offset

        # Open shared memory
        try:
            fd = os.open(f"/dev/shm/{shm_name}", os.O_RDONLY)
            self._shm_handle = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)
        except OSError as e:
            raise RuntimeError(f"Failed to attach shared memory: {e}")

    def detach_shared_memory(self) -> None:
        """Detach from shared memory."""
        if self._shm_handle is not None:
            self._shm_handle.close()
            self._shm_handle = None

    def execute(
        self,
        projection_output: np.ndarray,
        stream: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Execute the plugin: add delta to projection output.

        Args:
            projection_output: Output from linear projection
            stream: CUDA stream (optional)

        Returns:
            projection_output + delta
        """
        # Get delta from buffer or shared memory
        delta = self._get_delta(projection_output.shape)

        if delta is None:
            return projection_output

        # Add delta
        return projection_output + delta.astype(projection_output.dtype)

    def _get_delta(self, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """Get delta tensor."""
        if self._delta_buffer is not None:
            # Use provided buffer
            if self._delta_buffer.shape != shape:
                # Broadcast or slice as needed
                return self._delta_buffer[:shape[0], :shape[1], :shape[2]]
            return self._delta_buffer

        if self._shm_handle is not None:
            # Read from shared memory
            size = np.prod(shape) * 2  # float16 = 2 bytes
            self._shm_handle.seek(self.config.shm_offset)
            data = self._shm_handle.read(size)
            return np.frombuffer(data, dtype=np.float16).reshape(shape)

        return None

    def get_plugin_attrs(self) -> Dict[str, Any]:
        """Get plugin attributes for TensorRT registration."""
        return {
            'plugin_name': PLUGIN_NAME,
            'plugin_version': PLUGIN_VERSION,
            'plugin_namespace': PLUGIN_NAMESPACE,
            'layer_idx': self.config.layer_idx,
            'projection_type': self.config.projection_type,
            'hidden_size': self.config.hidden_size,
            'batch_size': self.config.batch_size,
        }


def create_trt_plugin_cpp_code() -> str:
    """
    Generate C++ code for the TensorRT plugin.

    This would be compiled into a shared library for production use.
    """
    return '''
// HELoRAProjectionPlugin.cpp
// TensorRT plugin for HE-LoRA delta injection

#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <string>

using namespace nvinfer1;

namespace helora {

class HELoRAProjectionPlugin : public IPluginV2DynamicExt {
public:
    HELoRAProjectionPlugin(int layerIdx, const char* projType, int hiddenSize)
        : mLayerIdx(layerIdx), mProjType(projType), mHiddenSize(hiddenSize) {}

    // IPluginV2 methods
    const char* getPluginType() const noexcept override { return "HELoRAProjection"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        return inputTypes[0];  // Same as input
    }

    // IPluginV2DynamicExt methods
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                   IExprBuilder& exprBuilder) noexcept override {
        return inputs[0];  // Same shape as input
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override {
        // Get delta from shared memory
        __half* delta = getDeltaFromShm();

        if (delta == nullptr) {
            // No delta, just copy input to output
            size_t bytes = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] *
                          inputDesc[0].dims.d[2] * sizeof(__half);
            cudaMemcpyAsync(outputs[0], inputs[0], bytes, cudaMemcpyDeviceToDevice, stream);
            return 0;
        }

        // Launch kernel: output = input + delta
        launchAddDeltaKernel(
            static_cast<const __half*>(inputs[0]),
            delta,
            static_cast<__half*>(outputs[0]),
            inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2],
            stream
        );

        return 0;
    }

    void setDeltaShmInfo(const char* shmName, size_t offset) {
        mShmName = shmName;
        mShmOffset = offset;
    }

private:
    int mLayerIdx;
    std::string mProjType;
    int mHiddenSize;
    std::string mShmName;
    size_t mShmOffset = 0;

    __half* getDeltaFromShm() {
        // Implementation would mmap shared memory and return pointer
        return nullptr;
    }

    void launchAddDeltaKernel(const __half* input, const __half* delta, __half* output,
                              size_t n, cudaStream_t stream);
};

// CUDA kernel for element-wise addition
__global__ void addDeltaKernel(const __half* input, const __half* delta, __half* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hadd(input[idx], delta[idx]);
    }
}

void HELoRAProjectionPlugin::launchAddDeltaKernel(
    const __half* input, const __half* delta, __half* output, size_t n, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addDeltaKernel<<<numBlocks, blockSize, 0, stream>>>(input, delta, output, n);
}

}  // namespace helora
'''


def create_plugin_build_script() -> str:
    """Generate CMake build script for the plugin."""
    return '''
# CMakeLists.txt for HE-LoRA TensorRT Plugin

cmake_minimum_required(VERSION 3.18)
project(HELoRAPlugin CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

add_library(helora_trt_plugin SHARED
    HELoRAProjectionPlugin.cpp
    HELoRAProjectionPlugin.cu
)

target_include_directories(helora_trt_plugin PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(helora_trt_plugin
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    nvinfer
)

set_target_properties(helora_trt_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
'''
