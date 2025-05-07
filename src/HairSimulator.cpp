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
    // gpu_data_ members are initialized by their default member initializers
}

HairSimulator::~HairSimulator() {
    free_gpu_memory();
}

void HairSimulator::unmap_and_delete_vbo() {
    if (gpu_data_.vbo_cuda_resource) {
        // Resource must be unmapped before unregistering.
        // cudaGraphicsUnregisterResource should handle unmapping if mapped by this context.
        cudaError_t err = cudaGraphicsUnregisterResource(gpu_data_.vbo_cuda_resource);
        if (err != cudaSuccess) {
            std::cerr << "HairSimulator: Warning - Failed to unregister VBO CUDA resource: " << cudaGetErrorString(err) << std::endl;
        }
        gpu_data_.vbo_cuda_resource = nullptr;
    }

    if (gpu_data_.vbo_id != 0) {
        // Requires an active OpenGL context
        glDeleteBuffers(1, &gpu_data_.vbo_id);
        GLenum glErr = glGetError();
        if (glErr != GL_NO_ERROR) {
            std::cerr << "HairSimulator: Warning - OpenGL error during glDeleteBuffers for VBO ID " << gpu_data_.vbo_id << ": " << reinterpret_cast<const char*>(glewGetErrorString(glErr)) << std::endl;
        }
        gpu_data_.vbo_id = 0;
    }
    gpu_data_.d_vbo_buffer_ptr = nullptr;
}

void HairSimulator::free_gpu_memory() {
    // Free standard CUDA particle data
    if (gpu_data_.d_posX) cudaFree(gpu_data_.d_posX);
    if (gpu_data_.d_posY) cudaFree(gpu_data_.d_posY);
    if (gpu_data_.d_posZ) cudaFree(gpu_data_.d_posZ);
    if (gpu_data_.d_strand_indices) cudaFree(gpu_data_.d_strand_indices);
    if (gpu_data_.d_particle_indices_in_strand) cudaFree(gpu_data_.d_particle_indices_in_strand);
    
    gpu_data_.d_posX = nullptr;
    gpu_data_.d_posY = nullptr;
    gpu_data_.d_posZ = nullptr;
    gpu_data_.d_strand_indices = nullptr;
    gpu_data_.d_particle_indices_in_strand = nullptr;

    // Free VBO related resources
    unmap_and_delete_vbo();
    gpu_data_.num_total_particles = 0;
}

void HairSimulator::upload_to_gpu(const std::vector<std::vector<float3>>& all_strands) {
    free_gpu_memory(); // Clear existing data and VBOs

    if (all_strands.empty()) {
        return;
    }

    for (const auto& strand : all_strands) {
        gpu_data_.num_total_particles += strand.size();
    }

    if (gpu_data_.num_total_particles == 0) {
        return; // No particles to upload
    }

    // Host SoA buffers
    std::vector<float> h_posX(gpu_data_.num_total_particles);
    std::vector<float> h_posY(gpu_data_.num_total_particles);
    std::vector<float> h_posZ(gpu_data_.num_total_particles);
    std::vector<int> h_strand_indices(gpu_data_.num_total_particles);
    std::vector<int> h_particle_indices_in_strand(gpu_data_.num_total_particles);

    int current_particle_idx = 0;
    for (int strand_idx = 0; strand_idx < all_strands.size(); ++strand_idx) {
        const auto& strand = all_strands[strand_idx];
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

    // Allocate GPU memory
    cudaError_t err;
    auto allocate_and_check = [&](void** devPtr, size_t size, const char* name) {
        err = cudaMalloc(devPtr, size);
        if (err != cudaSuccess) {
            free_gpu_memory();
            throw std::runtime_error("Failed to allocate " + std::string(name) + " on GPU: " + cudaGetErrorString(err));
        }
    };

    allocate_and_check(reinterpret_cast<void**>(&gpu_data_.d_posX), gpu_data_.num_total_particles * sizeof(float), "d_posX");
    allocate_and_check(reinterpret_cast<void**>(&gpu_data_.d_posY), gpu_data_.num_total_particles * sizeof(float), "d_posY");
    allocate_and_check(reinterpret_cast<void**>(&gpu_data_.d_posZ), gpu_data_.num_total_particles * sizeof(float), "d_posZ");
    allocate_and_check(reinterpret_cast<void**>(&gpu_data_.d_strand_indices), gpu_data_.num_total_particles * sizeof(int), "d_strand_indices");
    allocate_and_check(reinterpret_cast<void**>(&gpu_data_.d_particle_indices_in_strand), gpu_data_.num_total_particles * sizeof(int), "d_particle_indices_in_strand");

    // Copy data to GPU
    auto copy_and_check = [&](void* dst, const void* src, size_t size, const char* name) {
        err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            free_gpu_memory();
            throw std::runtime_error("Failed to copy " + std::string(name) + " to GPU: " + cudaGetErrorString(err));
        }
    };

    copy_and_check(gpu_data_.d_posX, h_posX.data(), gpu_data_.num_total_particles * sizeof(float), "posX");
    copy_and_check(gpu_data_.d_posY, h_posY.data(), gpu_data_.num_total_particles * sizeof(float), "posY");
    copy_and_check(gpu_data_.d_posZ, h_posZ.data(), gpu_data_.num_total_particles * sizeof(float), "posZ");
    copy_and_check(gpu_data_.d_strand_indices, h_strand_indices.data(), gpu_data_.num_total_particles * sizeof(int), "strand_indices");
    copy_and_check(gpu_data_.d_particle_indices_in_strand, h_particle_indices_in_strand.data(), gpu_data_.num_total_particles * sizeof(int), "particle_indices_in_strand");
}

void HairSimulator::create_and_populate_vbo() {
    if (gpu_data_.num_total_particles == 0) {
        std::cout << "HairSimulator: No particles to process. Skipping VBO creation." << std::endl;
        return;
    }
    if (!gpu_data_.d_posX || !gpu_data_.d_posY || !gpu_data_.d_posZ) {
         throw std::runtime_error("HairSimulator: Base particle position data (d_posX,Y,Z) not on GPU. Cannot create VBO.");
    }

    unmap_and_delete_vbo(); // Clean up any existing VBO

    cudaError_t cudaErr;
    GLenum glErr;
    size_t vbo_byte_size = gpu_data_.num_total_particles * 3 * sizeof(float);

    // 1. Create OpenGL VBO
    glGenBuffers(1, &gpu_data_.vbo_id);
    glErr = glGetError();
    if (glErr != GL_NO_ERROR || gpu_data_.vbo_id == 0) {
        unmap_and_delete_vbo();
        throw std::runtime_error("HairSimulator: Failed to generate OpenGL VBO. OpenGL Error: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(glErr))));
    }

    glBindBuffer(GL_ARRAY_BUFFER, gpu_data_.vbo_id);
    glBufferData(GL_ARRAY_BUFFER, vbo_byte_size, nullptr, GL_DYNAMIC_DRAW); // Allocate space, data from CUDA
    glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        unmap_and_delete_vbo();
        throw std::runtime_error("HairSimulator: Failed to allocate OpenGL VBO data (glBufferData). OpenGL Error: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(glErr))));
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 2. Register VBO with CUDA
    cudaErr = cudaGraphicsGLRegisterBuffer(&gpu_data_.vbo_cuda_resource, gpu_data_.vbo_id, cudaGraphicsRegisterFlagsWriteDiscard);
    if (cudaErr != cudaSuccess) {
        unmap_and_delete_vbo();
        throw std::runtime_error("HairSimulator: Failed to register VBO with CUDA: " + std::string(cudaGetErrorString(cudaErr)));
    }

    // 3. Map VBO for CUDA access
    cudaErr = cudaGraphicsMapResources(1, &gpu_data_.vbo_cuda_resource, 0); // Default stream
    if (cudaErr != cudaSuccess) {
        unmap_and_delete_vbo();
        throw std::runtime_error("HairSimulator: Failed to map VBO resource for CUDA: " + std::string(cudaGetErrorString(cudaErr)));
    }

    // 4. Get device pointer to mapped VBO memory
    size_t mapped_vbo_buffer_size;
    cudaErr = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&gpu_data_.d_vbo_buffer_ptr), &mapped_vbo_buffer_size, gpu_data_.vbo_cuda_resource);
    if (cudaErr != cudaSuccess || gpu_data_.d_vbo_buffer_ptr == nullptr) {
        cudaGraphicsUnmapResources(1, &gpu_data_.vbo_cuda_resource, 0); // Attempt unmap
        unmap_and_delete_vbo();
        throw std::runtime_error("HairSimulator: Failed to get mapped VBO pointer from CUDA: " + std::string(cudaGetErrorString(cudaErr)));
    }
    
    if (mapped_vbo_buffer_size < vbo_byte_size) {
        cudaGraphicsUnmapResources(1, &gpu_data_.vbo_cuda_resource, 0);
        unmap_and_delete_vbo();
        throw std::runtime_error("HairSimulator: Mapped VBO buffer size (" + std::to_string(mapped_vbo_buffer_size) +
                                 ") is smaller than expected (" + std::to_string(vbo_byte_size) + ").");
    }

    // 5. Launch CUDA kernel to populate VBO
    std::cout << "HairSimulator: Launching kernel to populate VBO ID: " << gpu_data_.vbo_id
              << " with " << gpu_data_.num_total_particles << " particles." << std::endl;
    
    launch_convert_pos_to_vbo_kernel(this->get_gpu_data(), gpu_data_.d_vbo_buffer_ptr);
    // Kernel itself should check for errors and synchronize.

    // 6. Unmap VBO, making it available to OpenGL
    cudaErr = cudaGraphicsUnmapResources(1, &gpu_data_.vbo_cuda_resource, 0);
    if (cudaErr != cudaSuccess) {
        unmap_and_delete_vbo(); 
        throw std::runtime_error("HairSimulator: Failed to unmap VBO resource after kernel: " + std::string(cudaGetErrorString(cudaErr)));
    }
    gpu_data_.d_vbo_buffer_ptr = nullptr; // Pointer is no longer valid for CUDA

    std::cout << "HairSimulator: VBO (ID: " << gpu_data_.vbo_id << ") populated successfully." << std::endl;
}

const HairGpuData& HairSimulator::get_gpu_data() const {
    return gpu_data_;
}
