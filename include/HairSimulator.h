#pragma once

#include <vector>
#include <cuda_runtime.h> // For float3 and CUDA API

struct HairGpuData {
    float *d_posX = nullptr;
    float *d_posY = nullptr;
    float *d_posZ = nullptr;
    int *d_strand_indices = nullptr;          // Index of the strand each particle belongs to
    int *d_particle_indices_in_strand = nullptr; // Index of the particle within its strand
    int num_total_particles = 0;
};

class HairSimulator
{
public:
    HairSimulator();
    ~HairSimulator();

    // Takes the strand data (AoS) and prepares SoA data on the GPU
    void upload_to_gpu(const std::vector<std::vector<float3>>& all_strands);

    // Getter for the GPU data structure
    const HairGpuData& get_gpu_data() const;

private:
    HairGpuData gpu_data_;

    // Helper to release GPU memory
    void free_gpu_memory();
};