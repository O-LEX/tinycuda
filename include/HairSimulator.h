#pragma once

#include <vector>
#include <cuda_runtime.h> // For float3 and CUDA API
#include <cuda_gl_interop.h> // For CUDA-OpenGL interop

struct HairGpuData {
    float *d_posX = nullptr;
    float *d_posY = nullptr;
    float *d_posZ = nullptr;
    int *d_strand_indices = nullptr;          // Index of the strand each particle belongs to
    int *d_particle_indices_in_strand = nullptr; // Index of the particle within its strand
    int num_total_particles = 0;

    // OpenGL VBO related data
    unsigned int vbo_id = 0; // OpenGL VBO ID
    float* d_vbo_buffer_ptr = nullptr; // CUDA device pointer to the mapped VBO buffer (valid only when mapped)
    struct cudaGraphicsResource *vbo_cuda_resource = nullptr; // CUDA graphics resource for VBO
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

    // Manages VBO creation, mapping to CUDA, and populating it with particle positions
    void create_and_populate_vbo();

private:
    HairGpuData gpu_data_;

    // Helper to release GPU memory (including VBO-related resources)
    void free_gpu_memory();
    // Helper to specifically handle VBO cleanup (unmapping, unregistering, deleting GL buffer)
    void unmap_and_delete_vbo();
};