#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "tiny.cuh"
#include "Shader.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA-OpenGL Interop Example", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    Shader shader("shader/vertex.glsl", "shader/fragment.glsl");

    
    const int numVertices = 1024;
    GLuint vbo, vao;
    struct cudaGraphicsResource* cudaVboResource;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window)) {
        float4* d_positions;
        size_t numBytes;
        cudaGraphicsMapResources(1, &cudaVboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &numBytes, cudaVboResource);

        float time = glfwGetTime();
        updateCuda(d_positions, numVertices, time);

        cudaGraphicsUnmapResources(1, &cudaVboResource, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        shader.use();
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, numVertices);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaVboResource);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}