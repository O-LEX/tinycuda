#include "dummy_kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h> // For printf in kernel

__global__ void dummy_cuda_kernel(
    const float* posX,
    const float* posY,
    const float* posZ,
    const int* strand_indices,
    const int* particle_indices_in_strand,
    int num_total_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_total_particles) {
        // Example: Print info for particle 0, only from thread 0 of block 0
        if (idx == 0) {
            printf("[CUDA dummy_kernel] Particle %d: Pos=(%f, %f, %f), StrandIdx=%d, ParticleInStrandIdx=%d\n",
                   idx,
                   posX[idx],
                   posY[idx],
                   posZ[idx],
                   strand_indices[idx],
                   particle_indices_in_strand[idx]);
        }
        // Dummy computation can be added here if needed
    }
}

void launch_dummy_kernel(const HairGpuData& gpu_data) { // Changed HairGpuSimData to gpu_data (HairGpuData type)
    if (gpu_data.num_total_particles == 0) {
        // printf("No particles to process in dummy_kernel.\n"); // Can be noisy
        return;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (gpu_data.num_total_particles + threads_per_block - 1) / threads_per_block;

    // printf("Launching dummy_cuda_kernel with %d blocks and %d threads for %d particles.\n",
    //        blocks_per_grid, threads_per_block, gpu_data.num_total_particles); // Can be noisy

    dummy_cuda_kernel<<<blocks_per_grid, threads_per_block>>>(
        gpu_data.d_posX,
        gpu_data.d_posY,
        gpu_data.d_posZ,
        gpu_data.d_strand_indices,
        gpu_data.d_particle_indices_in_strand,
        gpu_data.num_total_particles
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after launching dummy_cuda_kernel: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize(); // Wait and check for async errors
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during dummy_cuda_kernel execution: %s\n", cudaGetErrorString(err));
    }
    printf("dummy_cuda_kernel execution finished.\n"); // Keep this for confirmation
}

__global__ void convert_pos_to_vbo_kernel(
    const float* posX,
    const float* posY,
    const float* posZ,
    float* vbo_data, // Output: SoA (xyz, xyz, ...)
    int num_total_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_total_particles) {
        vbo_data[idx * 3 + 0] = posX[idx];
        vbo_data[idx * 3 + 1] = posY[idx];
        vbo_data[idx * 3 + 2] = posZ[idx];
    }
}

void launch_convert_pos_to_vbo_kernel(const HairGpuData& gpu_data, float* d_vbo_data) { // Changed HairGpuSimData to gpu_data (HairGpuData type)
    if (gpu_data.num_total_particles == 0) {
        // printf("No particles to process in convert_pos_to_vbo_kernel.\n"); // Can be noisy
        return;
    }
    if (d_vbo_data == nullptr) {
        fprintf(stderr, "Error: Output VBO data pointer (d_vbo_data) is null in launch_convert_pos_to_vbo_kernel.\n");
        return;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (gpu_data.num_total_particles + threads_per_block - 1) / threads_per_block;

    // printf("Launching convert_pos_to_vbo_kernel with %d blocks and %d threads for %d particles.\n",
    //        blocks_per_grid, threads_per_block, gpu_data.num_total_particles); // Can be noisy

    convert_pos_to_vbo_kernel<<<blocks_per_grid, threads_per_block>>>(
        gpu_data.d_posX,
        gpu_data.d_posY,
        gpu_data.d_posZ,
        d_vbo_data,
        gpu_data.num_total_particles
    );

    cudaError_t err = cudaGetLastError(); // Check for launch errors
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after launching convert_pos_to_vbo_kernel: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize(); // Wait for kernel to complete and check for asynchronous errors
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during convert_pos_to_vbo_kernel execution: %s\n", cudaGetErrorString(err));
    }
    printf("convert_pos_to_vbo_kernel execution finished.\n"); // Keep this for confirmation
}
