#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // For cudaGraphicsResource

// Forward declaration for HairSimulator to use HairGpuData
class HairSimulator;

// Pure CUDA simulation data
struct HairGpuData { // Renamed back from HairGpuSimData
    float *d_posX = nullptr;
    float *d_posY = nullptr;
    float *d_posZ = nullptr;
    int *d_strand_indices = nullptr;
    int *d_particle_indices_in_strand = nullptr;
    // Add other simulation-specific data pointers here if needed (e.g., velocity, mass)
    int num_total_particles = 0;
    // int num_total_strands = 0; // If needed for simulation logic
    // release() method removed as per user request
};

class HairSimulator
{
public:
    HairSimulator();
    ~HairSimulator();

    // Initializes simulation data on GPU and optionally sets up OpenGL interop for VBO
    bool initialize(const std::vector<std::vector<float3>>& raw_strands, bool setup_opengl_interop);

    // Maps the VBO for writing from CUDA and returns the device pointer.
    // Returns nullptr if not using OpenGL interop or if mapping fails.
    float* mapVboForWriting();

    // Unmaps the VBO after CUDA is done writing.
    void unmapVbo();

    // Provides read-only access to the simulation data.
    const HairGpuData& getSimulationData() const; // Changed from HairGpuSimData

    // Provides read-only access to the OpenGL VBO ID (for rendering).
    // Returns 0 if VBO is not initialized.
    unsigned int getVboId() const;

private:
    HairGpuData m_simData;          // CUDA simulation data (type changed from HairGpuSimData)
    bool m_useOpenGLInterop = false;

    // OpenGL Interop related resources (managed directly by HairSimulator)
    unsigned int m_vboId = 0;                           // OpenGL VBO ID
    struct cudaGraphicsResource *m_vboCudaResource = nullptr; // CUDA graphics resource for VBO

    // Helper to release all owned resources (CUDA memory and VBO-related resources)
    void releaseResources();
    
    // Internal helper for VBO cleanup (called by releaseResources)
    // Assumes OpenGL context is active if called for glDeleteBuffers
    void unmapAndUnregisterVbo(); 
    void deleteGLVbo(); 
};