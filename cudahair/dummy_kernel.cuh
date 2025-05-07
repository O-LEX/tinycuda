#pragma once
#include "HairSimulator.h" // For HairGpuData

// Wrapper function to launch the dummy CUDA kernel
void launch_dummy_kernel(const HairGpuData& gpuData);

// convert d_pos to vbo
void launch_convert_pos_to_vbo_kernel(const HairGpuData& gpuData, float* d_vbo_data);
