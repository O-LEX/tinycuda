// IMPORTANT: GLEW must be included before GLFW or other GL headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "HairSimulator.h"
#include "HairLoader.h"
#include "dummy_kernel.cuh"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

GLFWwindow* window = nullptr;

void init_opengl_context() {
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hidden window for offscreen operations
    window = glfwCreateWindow(100, 100, "Hidden OpenGL Context", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
}

void shutdown_opengl_context() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

int main() {
    init_opengl_context();

    { 
        HairLoader loader;
        loader.load("../data/strands00001.data");
        
        HairSimulator hair_sim;
        if (!hair_sim.initialize(loader.strands)) {
            shutdown_opengl_context();
            return 1;
        }

        const GpuData& sim_data_ref = hair_sim.getGpuData();

        if (sim_data_ref.num_total_particles > 0) {
            launch_dummy_kernel(sim_data_ref); 

            float* d_vbo_ptr = hair_sim.mapVbo();
            if (d_vbo_ptr) {
                launch_convert_pos_to_vbo_kernel(sim_data_ref, d_vbo_ptr);
                hair_sim.unmapVbo();
            } else if (sim_data_ref.num_total_particles > 0) { 
            }
        } else {
        }
    } 

    shutdown_opengl_context();
    return 0;
}
