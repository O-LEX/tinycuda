#pragma once
#include "HairDataManager.h" // For HairGpuData

// Wrapper function to launch the dummy CUDA kernel
void launch_dummy_kernel(const HairGpuData& gpuData);
