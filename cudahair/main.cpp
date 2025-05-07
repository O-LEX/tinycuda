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

    { // Scope for HairLoader and HairSimulator to ensure destructors are called before OpenGL shutdown
        HairLoader loader;
        try {
            // Adjust path as needed.
            loader.load("../data/strands00001.data");
            std::cout << "Successfully loaded hair data. Number of strands: " << loader.strands.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading hair data: " << e.what() << std::endl;
            shutdown_opengl_context();
            return 1;
        }

        HairSimulator hair_sim;
        try {
            hair_sim.upload_to_gpu(loader.strands);
            const HairGpuData& gpu_data_ref = hair_sim.get_gpu_data();
            std::cout << "Successfully uploaded hair particle data to GPU. Total particles: " << gpu_data_ref.num_total_particles << std::endl;

            if (gpu_data_ref.num_total_particles > 0) {
                // Optional: Display first particle's data for verification
                // std::vector<float> posX(1), posY(1), posZ(1);
                // std::vector<int> strandIdx(1), particleInStrandIdx(1);
                // cudaMemcpy(posX.data(), gpu_data_ref.d_posX, sizeof(float), cudaMemcpyDeviceToHost);
                // cudaMemcpy(posY.data(), gpu_data_ref.d_posY, sizeof(float), cudaMemcpyDeviceToHost);
                // cudaMemcpy(posZ.data(), gpu_data_ref.d_posZ, sizeof(float), cudaMemcpyDeviceToHost);
                // cudaMemcpy(strandIdx.data(), gpu_data_ref.d_strand_indices, sizeof(int), cudaMemcpyDeviceToHost);
                // cudaMemcpy(particleInStrandIdx.data(), gpu_data_ref.d_particle_indices_in_strand, sizeof(int), cudaMemcpyDeviceToHost);
                // std::cout << "First particle data on GPU (host copy):\n"
                //           << "  Position: (" << posX[0] << ", " << posY[0] << ", " << posZ[0] << ")\n"
                //           << "  Strand Index: " << strandIdx[0] << "\n"
                //           << "  Particle Index in Strand: " << particleInStrandIdx[0] << std::endl;

                std::cout << "\nLaunching dummy CUDA kernel..." << std::endl;
                launch_dummy_kernel(gpu_data_ref);
                // The dummy kernel now prints its own completion message.

                std::cout << "\nAttempting to create and populate VBO via CUDA-OpenGL interop..." << std::endl;
                hair_sim.create_and_populate_vbo();

                if (gpu_data_ref.vbo_id != 0) {
                    std::cout << "OpenGL VBO (ID: " << gpu_data_ref.vbo_id << ") created and populated successfully." << std::endl;
                    std::cout << "This VBO contains " << gpu_data_ref.num_total_particles << " particle positions (XYZ format)." << std::endl;
                } else if (gpu_data_ref.num_total_particles > 0) {
                    std::cerr << "VBO creation/population failed or was skipped." << std::endl;
                }
            } else {
                std::cout << "No particles loaded; skipping CUDA kernel launch and VBO creation." << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error during HairSimulator operations or kernel launch: " << e.what() << std::endl;
            // hair_sim destructor called automatically at scope end
            shutdown_opengl_context();
            return 1;
        }

        std::cout << "\nHair simulation test completed." << std::endl;
        // hair_sim destructor called here as it goes out of scope
    } // End of scope for HairLoader and HairSimulator

    shutdown_opengl_context();
    return 0;
}
