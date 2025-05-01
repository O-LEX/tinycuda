#include "tiny.cuh"

__global__ void modifyVertices(float4* positions, int numVertices, float time) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numVertices) {
        float x = (idx / (float)numVertices) * 2.0f - 1.0f; // Map index to [-1, 1]
        float y = sinf(2.0f * 3.14159f * (x - time)); // Sine wave moving left
        positions[idx].x = x;
        positions[idx].y = y * 0.5f; // Scale amplitude
        positions[idx].z = 0.0f;
        positions[idx].w = 1.0f;
    }
}

void updateCuda(float4* positions, int numVertices, float time) {
    modifyVertices<<<(numVertices + 255) / 256, 256>>>(positions, numVertices, time);
}
