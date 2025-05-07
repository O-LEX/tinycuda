#include "HairDataManager.h"
#include "HairLoader.h"
#include "dummy_kernel.cuh" // Include the dummy kernel header
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main() {
    // Create HairLoader instance
    HairLoader loader;

    // Load hair data
    try {
        loader.load("../data/strands00001.data"); // Adjust path as needed
        std::cout << "Successfully loaded hair data. Number of strands: " << loader.strands.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading hair data: " << e.what() << std::endl;
        return 1;
    }

    // Create HairDataManager instance
    HairDataManager dataManager;

    // Upload data to GPU
    try {
        dataManager.upload_to_gpu(loader.strands);
        const HairGpuData& gpuData = dataManager.get_gpu_data();
        std::cout << "Successfully uploaded data to GPU. Total particles: " << gpuData.num_total_particles << std::endl;

        // Example: Print out the first particle's data if it exists
        if (gpuData.num_total_particles > 0) {
            std::vector<float> posX(1);
            std::vector<float> posY(1);
            std::vector<float> posZ(1);
            std::vector<int> strandIdx(1);
            std::vector<int> particleInStrandIdx(1);

            cudaError_t err;
            err = cudaMemcpy(posX.data(), gpuData.d_posX, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy d_posX for first particle");
            err = cudaMemcpy(posY.data(), gpuData.d_posY, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy d_posY for first particle");
            err = cudaMemcpy(posZ.data(), gpuData.d_posZ, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy d_posZ for first particle");
            err = cudaMemcpy(strandIdx.data(), gpuData.d_strand_indices, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy d_strand_indices for first particle");
            err = cudaMemcpy(particleInStrandIdx.data(), gpuData.d_particle_indices_in_strand, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy d_particle_indices_in_strand for first particle");

            std::cout << "First particle data on GPU:" << std::endl;
            std::cout << "  Position: (" << posX[0] << ", " << posY[0] << ", " << posZ[0] << ")" << std::endl;
            std::cout << "  Strand Index: " << strandIdx[0] << std::endl;
            std::cout << "  Particle Index in Strand: " << particleInStrandIdx[0] << std::endl;
        }

        // Launch the dummy CUDA kernel
        std::cout << "\nLaunching CUDA kernel..." << std::endl;
        launch_dummy_kernel(gpuData);
        std::cout << "CUDA kernel launch sequence finished.\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error in HairDataManager or Kernel Launch: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "HairDataManager test with dummy kernel completed successfully." << std::endl;
    return 0;
}
