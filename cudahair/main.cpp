#include "HairSimulator.h"
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
    HairSimulator dataManager;

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

        // --- Test VBO Conversion Kernel ---
        if (gpuData.num_total_particles > 0) {
            float* d_vbo_data = nullptr;
            size_t vbo_size = gpuData.num_total_particles * 3 * sizeof(float);
            cudaError_t err_alloc = cudaMalloc(reinterpret_cast<void**>(&d_vbo_data), vbo_size);
            if (err_alloc != cudaSuccess) {
                std::cerr << "Failed to allocate d_vbo_data on GPU: " << cudaGetErrorString(err_alloc) << std::endl;
            } else {
                std::cout << "\nLaunching VBO conversion kernel..." << std::endl;
                launch_convert_pos_to_vbo_kernel(gpuData, d_vbo_data);
                std::cout << "VBO conversion kernel launch sequence finished.\n" << std::endl;

                // Optional: Verify by copying some data back
                int num_particles_to_check = std::min(5, gpuData.num_total_particles);
                if (num_particles_to_check > 0) {
                    std::vector<float> h_vbo_data_check(num_particles_to_check * 3);
                    cudaError_t err_copy = cudaMemcpy(h_vbo_data_check.data(), d_vbo_data, num_particles_to_check * 3 * sizeof(float), cudaMemcpyDeviceToHost);
                    if (err_copy != cudaSuccess) {
                        std::cerr << "Failed to copy VBO data back to host for verification: " << cudaGetErrorString(err_copy) << std::endl;
                    } else {
                        std::cout << "\nFirst " << num_particles_to_check << " particles in VBO format (on host):" << std::endl;
                        for (int i = 0; i < num_particles_to_check; ++i) {
                            std::cout << "  Particle " << i << ": ("
                                      << h_vbo_data_check[i * 3 + 0] << ", "
                                      << h_vbo_data_check[i * 3 + 1] << ", "
                                      << h_vbo_data_check[i * 3 + 2] << ")" << std::endl;
                        }
                    }
                }
                cudaFree(d_vbo_data);
                std::cout << "Freed d_vbo_data." << std::endl;
            }
        }
        // --- End Test VBO Conversion Kernel ---

    } catch (const std::exception& e) {
        std::cerr << "Error in HairDataManager or Kernel Launch: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "HairDataManager test with dummy kernel completed successfully." << std::endl;
    return 0;
}
