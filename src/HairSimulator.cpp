// IMPORTANT: GLEW must be included before any other header that might include gl.h (like cuda_gl_interop.h or GLFW)
#include <GL/glew.h>

#include "HairSimulator.h"
#include "dummy_kernel.cuh" // For launch_convert_pos_to_vbo_kernel

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // For CUDA-OpenGL interop functions
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

// Note: OpenGL functions (glGenBuffers, etc.) require an active OpenGL context
// and GLEW to be initialized. This is handled in main.cpp.

HairSimulator::HairSimulator() {
    // Members are initialized by their default initializers (nullptr, 0, false)
}

HairSimulator::~HairSimulator() {
    releaseResources();
}

void HairSimulator::releaseResources() {
    // Release CUDA simulation buffers directly as HairGpuData no longer has a release() method
    if (m_simData.d_posX) { cudaFree(m_simData.d_posX); m_simData.d_posX = nullptr; }
    if (m_simData.d_posY) { cudaFree(m_simData.d_posY); m_simData.d_posY = nullptr; }
    if (m_simData.d_posZ) { cudaFree(m_simData.d_posZ); m_simData.d_posZ = nullptr; }
    if (m_simData.d_strand_indices) { cudaFree(m_simData.d_strand_indices); m_simData.d_strand_indices = nullptr; }
    if (m_simData.d_particle_indices_in_strand) { cudaFree(m_simData.d_particle_indices_in_strand); m_simData.d_particle_indices_in_strand = nullptr; }
    m_simData.num_total_particles = 0;

    unmapAndUnregisterVbo();
    deleteGLVbo(); 
}

void HairSimulator::unmapAndUnregisterVbo() {
    if (m_vboCudaResource) {
        // According to CUDA docs, resource should be unmapped before unregistering.
        // cudaGraphicsUnmapResources might have been called in unmapVbo(), but an extra call is safe.
        // cudaGraphicsUnregisterResource itself should handle unmapping if the resource was mapped by this context.
        cudaError_t err_unmap = cudaGraphicsUnmapResources(1, &m_vboCudaResource, 0); // Attempt unmap just in case
        if (err_unmap != cudaSuccess && err_unmap != cudaErrorNotMapped) { // Ignore if not mapped
             std::cerr << "HairSimulator: Warning - Failed to unmap VBO resource during unregister: " << cudaGetErrorString(err_unmap) << std::endl;
        }

        cudaError_t err_unregister = cudaGraphicsUnregisterResource(m_vboCudaResource);
        if (err_unregister != cudaSuccess) {
            std::cerr << "HairSimulator: Warning - Failed to unregister VBO CUDA resource: " << cudaGetErrorString(err_unregister) << std::endl;
        }
        m_vboCudaResource = nullptr;
    }
}

void HairSimulator::deleteGLVbo() {
    if (m_vboId != 0) {
        // IMPORTANT: Requires an active OpenGL context
        glDeleteBuffers(1, &m_vboId);
        GLenum glErr = glGetError();
        if (glErr != GL_NO_ERROR) {
            std::cerr << "HairSimulator: Warning - OpenGL error during glDeleteBuffers for VBO ID " << m_vboId << ": " << reinterpret_cast<const char*>(glewGetErrorString(glErr)) << std::endl;
        }
        m_vboId = 0;
    }
}


bool HairSimulator::initialize(const std::vector<std::vector<float3>>& raw_strands, bool setup_opengl_interop) {
    releaseResources(); // Clear any existing data

    m_useOpenGLInterop = setup_opengl_interop;

    if (raw_strands.empty()) {
        return true; // Successfully initialized with no data
    }

    // 1. Calculate total particles and prepare host SoA data
    for (const auto& strand : raw_strands) {
        m_simData.num_total_particles += strand.size();
    }

    if (m_simData.num_total_particles == 0) {
        return true; // Successfully initialized with no particles
    }

    std::vector<float> h_posX(m_simData.num_total_particles);
    std::vector<float> h_posY(m_simData.num_total_particles);
    std::vector<float> h_posZ(m_simData.num_total_particles);
    std::vector<int> h_strand_indices(m_simData.num_total_particles);
    std::vector<int> h_particle_indices_in_strand(m_simData.num_total_particles);

    int current_particle_idx = 0;
    for (int strand_idx = 0; strand_idx < raw_strands.size(); ++strand_idx) {
        const auto& strand = raw_strands[strand_idx];
        for (int particle_in_strand_idx = 0; particle_in_strand_idx < strand.size(); ++particle_in_strand_idx) {
            const auto& particle = strand[particle_in_strand_idx];
            h_posX[current_particle_idx] = particle.x;
            h_posY[current_particle_idx] = particle.y;
            h_posZ[current_particle_idx] = particle.z;
            h_strand_indices[current_particle_idx] = strand_idx;
            h_particle_indices_in_strand[current_particle_idx] = particle_in_strand_idx;
            current_particle_idx++;
        }
    }

    // 2. Allocate GPU memory for simulation data and copy from host
    cudaError_t err;
    auto allocate_and_copy = [&](void** devPtr, const void* hostPtr, size_t size, const char* name) {
        err = cudaMalloc(devPtr, size);
        if (err != cudaSuccess) {
            releaseResources(); // Cleanup partially allocated resources
            throw std::runtime_error("Failed to allocate " + std::string(name) + " on GPU: " + cudaGetErrorString(err));
        }
        err = cudaMemcpy(*devPtr, hostPtr, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            releaseResources();
            throw std::runtime_error("Failed to copy " + std::string(name) + " to GPU: " + cudaGetErrorString(err));
        }
    };

    try {
        allocate_and_copy(reinterpret_cast<void**>(&m_simData.d_posX), h_posX.data(), m_simData.num_total_particles * sizeof(float), "d_posX");
        allocate_and_copy(reinterpret_cast<void**>(&m_simData.d_posY), h_posY.data(), m_simData.num_total_particles * sizeof(float), "d_posY");
        allocate_and_copy(reinterpret_cast<void**>(&m_simData.d_posZ), h_posZ.data(), m_simData.num_total_particles * sizeof(float), "d_posZ");
        allocate_and_copy(reinterpret_cast<void**>(&m_simData.d_strand_indices), h_strand_indices.data(), m_simData.num_total_particles * sizeof(int), "d_strand_indices");
        allocate_and_copy(reinterpret_cast<void**>(&m_simData.d_particle_indices_in_strand), h_particle_indices_in_strand.data(), m_simData.num_total_particles * sizeof(int), "d_particle_indices_in_strand");
    } catch (const std::exception& e) {
        // releaseResources() is called by the helper on error, so just rethrow or handle
        std::cerr << "Error during GPU data allocation/copy: " << e.what() << std::endl;
        return false; // Indicate initialization failure
    }

    // 3. If using OpenGL interop, create VBO and register with CUDA
    if (m_useOpenGLInterop && m_simData.num_total_particles > 0) {
        GLenum glErr;
        size_t vbo_byte_size = m_simData.num_total_particles * 3 * sizeof(float);

        glGenBuffers(1, &m_vboId);
        glErr = glGetError();
        if (glErr != GL_NO_ERROR || m_vboId == 0) {
            releaseResources();
            throw std::runtime_error("HairSimulator: Failed to generate OpenGL VBO. OpenGL Error: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(glErr))));
        }

        glBindBuffer(GL_ARRAY_BUFFER, m_vboId);
        glBufferData(GL_ARRAY_BUFFER, vbo_byte_size, nullptr, GL_DYNAMIC_DRAW);
        glErr = glGetError();
        if (glErr != GL_NO_ERROR) {
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            releaseResources();
            throw std::runtime_error("HairSimulator: Failed to allocate OpenGL VBO data (glBufferData). OpenGL Error: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(glErr))));
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        err = cudaGraphicsGLRegisterBuffer(&m_vboCudaResource, m_vboId, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess) {
            releaseResources();
            throw std::runtime_error("HairSimulator: Failed to register VBO with CUDA: " + std::string(cudaGetErrorString(err)));
        }
    }
    return true; // Successfully initialized
}

float* HairSimulator::mapVboForWriting() {
    if (!m_useOpenGLInterop || !m_vboCudaResource || m_simData.num_total_particles == 0) {
        return nullptr;
    }

    cudaError_t cudaErr = cudaGraphicsMapResources(1, &m_vboCudaResource, 0);
    if (cudaErr != cudaSuccess) {
        std::cerr << "HairSimulator: Failed to map VBO resource for CUDA: " << cudaGetErrorString(cudaErr) << std::endl;
        return nullptr;
    }

    float* d_vbo_ptr = nullptr;
    size_t mapped_vbo_buffer_size;
    cudaErr = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_vbo_ptr), &mapped_vbo_buffer_size, m_vboCudaResource);
    if (cudaErr != cudaSuccess || d_vbo_ptr == nullptr) {
        cudaGraphicsUnmapResources(1, &m_vboCudaResource, 0); // Attempt to unmap before returning
        std::cerr << "HairSimulator: Failed to get mapped VBO pointer from CUDA: " << cudaGetErrorString(cudaErr) << std::endl;
        return nullptr;
    }

    size_t expected_vbo_byte_size = m_simData.num_total_particles * 3 * sizeof(float);
    if (mapped_vbo_buffer_size < expected_vbo_byte_size) {
        cudaGraphicsUnmapResources(1, &m_vboCudaResource, 0);
        std::cerr << "HairSimulator: Mapped VBO buffer size (" << mapped_vbo_buffer_size
                  << ") is smaller than expected (" << expected_vbo_byte_size << ")." << std::endl;
        return nullptr;
    }
    return d_vbo_ptr;
}

void HairSimulator::unmapVbo() {
    if (!m_useOpenGLInterop || !m_vboCudaResource) {
        return;
    }
    cudaError_t cudaErr = cudaGraphicsUnmapResources(1, &m_vboCudaResource, 0);
    if (cudaErr != cudaSuccess) {
        // This can be a critical error if data was being written, but for now, just log it.
        std::cerr << "HairSimulator: Warning - Failed to unmap VBO resource: " << cudaGetErrorString(cudaErr) << std::endl;
    }
}

const HairGpuData& HairSimulator::getSimulationData() const {
    return m_simData;
}

unsigned int HairSimulator::getVboId() const {
    return m_vboId;
}

// Old methods to be removed or adapted:
// void HairSimulator::upload_to_gpu(const std::vector<std::vector<float3>>& all_strands) { ... } // Replaced by initialize()
// void HairSimulator::create_and_populate_vbo() { ... } // Integrated into initialize() and map/unmap logic
// const HairGpuData& HairSimulator::get_gpu_data() const { ... } // Replaced by getSimulationData()
// void HairSimulator::free_gpu_memory() { ... } // Replaced by releaseResources()
// void HairSimulator::unmap_and_delete_vbo() { ... } // Replaced by unmapAndUnregisterVbo() and deleteGLVbo()
