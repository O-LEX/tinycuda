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
            printf("[CUDA Kernel] Particle %d: Pos=(%f, %f, %f), StrandIdx=%d, ParticleInStrandIdx=%d\n",
                   idx,
                   posX[idx],
                   posY[idx],
                   posZ[idx],
                   strand_indices[idx],
                   particle_indices_in_strand[idx]);
        }
        // Add any other dummy computation here if needed
    }
}

void launch_dummy_kernel(const HairGpuData& gpuData) {
    if (gpuData.num_total_particles == 0) {
        printf("No particles to process in dummy_kernel.\n");
        return;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (gpuData.num_total_particles + threads_per_block - 1) / threads_per_block;

    printf("Launching dummy_cuda_kernel with %d blocks and %d threads per block for %d particles.\n",
           blocks_per_grid, threads_per_block, gpuData.num_total_particles);

    dummy_cuda_kernel<<<blocks_per_grid, threads_per_block>>>(
        gpuData.d_posX,
        gpuData.d_posY,
        gpuData.d_posZ,
        gpuData.d_strand_indices,
        gpuData.d_particle_indices_in_strand,
        gpuData.num_total_particles
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after launching dummy_cuda_kernel: %s\n", cudaGetErrorString(err));
        // It's often good to exit or throw an exception here in real applications
    }

    // Wait for the kernel to complete and check for any asynchronous errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during dummy_cuda_kernel execution: %s\n", cudaGetErrorString(err));
    }
    printf("dummy_cuda_kernel execution finished.\n");
}
