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
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hidden window for offscreen operations
    window = glfwCreateWindow(100, 100, "Hidden OpenGL Context", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window or OpenGL context" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    std::cout << "OpenGL context and GLEW initialized successfully." << std::endl;
    std::cout << "OpenGL Version: " << reinterpret_cast<const char*>(glGetString(GL_VERSION)) << std::endl;
}

void shutdown_opengl_context() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
    std::cout << "OpenGL context shut down." << std::endl;
}

int main() {
    std::cout << "Starting Hair Simulation Test Program..." << std::endl;

    init_opengl_context();

    { 
        HairLoader loader;
        try {
            loader.load("../data/strands00001.data");
            std::cout << "Successfully loaded hair data. Number of strands: " << loader.strands.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading hair data: " << e.what() << std::endl;
            shutdown_opengl_context();
            return 1;
        }

        HairSimulator hair_sim;
        try {
            // Initialize simulator (OpenGL interop is now always enabled)
            if (!hair_sim.initialize(loader.strands)) { // Second parameter removed
                std::cerr << "Failed to initialize HairSimulator." << std::endl;
                shutdown_opengl_context();
                return 1;
            }

            const HairGpuData& sim_data_ref = hair_sim.getSimulationData();
            std::cout << "Successfully initialized hair particle data on GPU. Total particles: " << sim_data_ref.num_total_particles << std::endl;

            if (sim_data_ref.num_total_particles > 0) {
                std::cout << "\nLaunching dummy CUDA kernel..." << std::endl;
                launch_dummy_kernel(sim_data_ref); 

                std::cout << "\nAttempting to populate VBO via CUDA-OpenGL interop..." << std::endl;
                float* d_vbo_ptr = hair_sim.mapVboForWriting();
                if (d_vbo_ptr) {
                    std::cout << "HairSimulator: Launching kernel to populate VBO ID: " << hair_sim.getVboId()
                              << " with " << sim_data_ref.num_total_particles << " particles." << std::endl;
                    launch_convert_pos_to_vbo_kernel(sim_data_ref, d_vbo_ptr);
                    hair_sim.unmapVbo();
                    std::cout << "HairSimulator: VBO (ID: " << hair_sim.getVboId() << ") populated successfully." << std::endl;

                    if (hair_sim.getVboId() != 0) {
                        std::cout << "OpenGL VBO (ID: " << hair_sim.getVboId() << ") created and populated successfully." << std::endl;
                        std::cout << "This VBO contains " << sim_data_ref.num_total_particles << " particle positions (XYZ format)." << std::endl;
                    }
                } else if (sim_data_ref.num_total_particles > 0) { // Check if mapping was expected
                    std::cerr << "VBO mapping failed or was skipped unexpectedly." << std::endl;
                }
            } else {
                std::cout << "No particles loaded; skipping CUDA kernel launch and VBO creation." << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error during HairSimulator operations or kernel launch: " << e.what() << std::endl;
            shutdown_opengl_context();
            return 1;
        }

        std::cout << "\nHair simulation test completed." << std::endl;
    } 

    shutdown_opengl_context();
    return 0;
}
