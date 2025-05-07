#include "HairDataManager.h"
#include <cuda_runtime.h> // For cudaMalloc, cudaMemcpy, cudaFree, cudaError_t, cudaGetErrorString
#include <stdexcept>      // For std::runtime_error
#include <vector>         // For std::vector on host side for SoA conversion
#include <string>         // For std::string in error messages

HairDataManager::HairDataManager() {
    // gpu_data_ members are initialized by their default member initializers (nullptr, 0)
}

HairDataManager::~HairDataManager() {
    free_gpu_memory();
}

void HairDataManager::free_gpu_memory() {
    if (gpu_data_.d_posX) {
        cudaFree(gpu_data_.d_posX);
        gpu_data_.d_posX = nullptr;
    }
    if (gpu_data_.d_posY) {
        cudaFree(gpu_data_.d_posY);
        gpu_data_.d_posY = nullptr;
    }
    if (gpu_data_.d_posZ) {
        cudaFree(gpu_data_.d_posZ);
        gpu_data_.d_posZ = nullptr;
    }
    if (gpu_data_.d_strand_indices) {
        cudaFree(gpu_data_.d_strand_indices);
        gpu_data_.d_strand_indices = nullptr;
    }
    if (gpu_data_.d_particle_indices_in_strand) {
        cudaFree(gpu_data_.d_particle_indices_in_strand);
        gpu_data_.d_particle_indices_in_strand = nullptr;
    }
    gpu_data_.num_total_particles = 0;
}

void HairDataManager::upload_to_gpu(const std::vector<std::vector<float3>>& all_strands) {
    // Free any existing memory first to prevent leaks if called multiple times
    free_gpu_memory();

    if (all_strands.empty()) {
        // Nothing to upload
        return;
    }

    // 1. Calculate total number of particles
    for (const auto& strand : all_strands) {
        gpu_data_.num_total_particles += strand.size();
    }

    if (gpu_data_.num_total_particles == 0) {
        // All strands are empty, or no strands with particles
        return;
    }

    // 2. Allocate temporary host memory for SoA conversion and mapping
    std::vector<float> h_posX(gpu_data_.num_total_particles);
    std::vector<float> h_posY(gpu_data_.num_total_particles);
    std::vector<float> h_posZ(gpu_data_.num_total_particles);
    std::vector<int> h_strand_indices(gpu_data_.num_total_particles);
    std::vector<int> h_particle_indices_in_strand(gpu_data_.num_total_particles);

    // 3. Populate host SoA arrays and mapping arrays by iterating through AoS data
    int current_particle_idx = 0;
    for (int strand_idx = 0; strand_idx < all_strands.size(); ++strand_idx) {
        const auto& strand = all_strands[strand_idx];
        for (int particle_in_strand_idx = 0; particle_in_strand_idx < strand.size(); ++particle_in_strand_idx) {
            const auto& particle = strand[particle_in_strand_idx];
            if (current_particle_idx < gpu_data_.num_total_particles) {
                h_posX[current_particle_idx] = particle.x;
                h_posY[current_particle_idx] = particle.y;
                h_posZ[current_particle_idx] = particle.z;
                h_strand_indices[current_particle_idx] = strand_idx;
                h_particle_indices_in_strand[current_particle_idx] = particle_in_strand_idx;
                current_particle_idx++;
            } else {
                throw std::runtime_error("Particle index out of bounds during SoA conversion. This indicates a logic error.");
            }
        }
    }

    // 4. Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&gpu_data_.d_posX), gpu_data_.num_total_particles * sizeof(float));
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to allocate d_posX on GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(reinterpret_cast<void**>(&gpu_data_.d_posY), gpu_data_.num_total_particles * sizeof(float));
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to allocate d_posY on GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(reinterpret_cast<void**>(&gpu_data_.d_posZ), gpu_data_.num_total_particles * sizeof(float));
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to allocate d_posZ on GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(reinterpret_cast<void**>(&gpu_data_.d_strand_indices), gpu_data_.num_total_particles * sizeof(int));
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to allocate d_strand_indices on GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(reinterpret_cast<void**>(&gpu_data_.d_particle_indices_in_strand), gpu_data_.num_total_particles * sizeof(int));
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to allocate d_particle_indices_in_strand on GPU: " + std::string(cudaGetErrorString(err)));
    }

    // 5. Copy data from host SoA and mapping arrays to GPU
    err = cudaMemcpy(gpu_data_.d_posX, h_posX.data(), gpu_data_.num_total_particles * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to copy posX to GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(gpu_data_.d_posY, h_posY.data(), gpu_data_.num_total_particles * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to copy posY to GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(gpu_data_.d_posZ, h_posZ.data(), gpu_data_.num_total_particles * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to copy posZ to GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(gpu_data_.d_strand_indices, h_strand_indices.data(), gpu_data_.num_total_particles * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to copy strand_indices to GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(gpu_data_.d_particle_indices_in_strand, h_particle_indices_in_strand.data(), gpu_data_.num_total_particles * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_gpu_memory();
        throw std::runtime_error("Failed to copy particle_indices_in_strand to GPU: " + std::string(cudaGetErrorString(err)));
    }
}

const HairGpuData& HairDataManager::get_gpu_data() const {
    return gpu_data_;
}
