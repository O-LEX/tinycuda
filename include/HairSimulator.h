#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // For cudaGraphicsResource

// Forward declaration for HairSimulator to use HairGpuData
class HairSimulator;

struct GpuData {
    float *d_posX = nullptr;
    float *d_posY = nullptr;
    float *d_posZ = nullptr;
    int *d_strand_indices = nullptr;
    int *d_particle_indices_in_strand = nullptr;
    int num_total_particles = 0;
};

class HairSimulator {
public:
    HairSimulator();
    ~HairSimulator();

    bool initialize(const std::vector<std::vector<float3>>& raw_strands);

    float* mapVbo();
    void unmapVbo();

    const GpuData& getGpuData() const;
    unsigned int getVboId() const;

private:
    GpuData m_GpuData;

    // OpenGL Interop related resources (managed directly by HairSimulator)
    unsigned int m_vboId = 0;                           // OpenGL VBO ID
    struct cudaGraphicsResource *m_vboCudaResource = nullptr; // CUDA graphics resource for VBO

    // Helper to release all owned resources (CUDA memory and VBO-related resources)
    void releaseResources();
    void deleteVbo(); 
};