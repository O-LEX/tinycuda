#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__global__ void modifyVertices(float4* positions, int numVertices, float time);

void updateCuda(float4 *positions, int numVertices, float time);