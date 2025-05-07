#pragma once
#include "HairSimulator.h" // For HairGpuData

// Wrapper function to launch the dummy CUDA kernel
void launch_dummy_kernel(const GpuData& gpu_data);

// convert d_pos to vbo
void launch_convert_pos_to_vbo_kernel(const GpuData& gpu_data, float* d_vbo_data);
