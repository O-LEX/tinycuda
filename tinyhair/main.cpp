#define _USE_MATH_DEFINES // For M_PI
#include <cmath> // For cosf, sinf, sqrtf, tanf, M_PI
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include "HairLoader.h"
#include "Shader.h"
#include <cuda_runtime.h> // Required for float3 definition
#include "helper_math.h" // Explicitly include for vector operations
#include <vector>
#include <limits> // For std::numeric_limits
#include <numeric> // For std::iota and std::numeric_limits

// Basic camera parameters (adjust as needed)
float cameraYaw = -90.0f;
float cameraPitch = 0.0f;
float cameraZoom = 45.0f; // Perspective FOV
float lastX = 400, lastY = 300; // Initial mouse position
bool firstMouse = true;

// Camera position
float3 cameraPos = make_float3(0.0f, 0.0f, 50.0f); // Move camera further back
float3 cameraFront = make_float3(0.0f, 0.0f, -1.0f);
float3 cameraUp = make_float3(0.0f, 1.0f, 0.0f);

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    cameraYaw += xoffset;
    cameraPitch += yoffset;

    // Clamp pitch
    if (cameraPitch > 89.0f)
        cameraPitch = 89.0f;
    if (cameraPitch < -89.0f)
        cameraPitch = -89.0f;

    float3 front;
    front.x = cosf(cameraYaw * M_PI / 180.0f) * cosf(cameraPitch * M_PI / 180.0f);
    front.y = sinf(cameraPitch * M_PI / 180.0f);
    front.z = sinf(cameraYaw * M_PI / 180.0f) * cosf(cameraPitch * M_PI / 180.0f);
    // Normalize is missing in helper_math.h for float3, implement or use component-wise ops
    float len = sqrtf(front.x*front.x + front.y*front.y + front.z*front.z);
    cameraFront = make_float3(front.x/len, front.y/len, front.z/len);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    cameraZoom -= (float)yoffset;
    if (cameraZoom < 1.0f) cameraZoom = 1.0f;
    if (cameraZoom > 45.0f) cameraZoom = 45.0f;
}

// Simple LookAt matrix calculation (replace with Eigen or GLM if available/preferred)
void calculate_lookat(float* matrix, float3 eye, float3 center, float3 up) {
    // Note: Ensure helper_math.h functions are correctly used
    float3 f = normalize(center - eye); // operator- and normalize are from helper_math.h
    float3 s = normalize(cross(f, up)); // cross and normalize are from helper_math.h
    float3 u = cross(s, f); // cross is from helper_math.h

    matrix[0] = s.x;
    matrix[1] = u.x;
    matrix[2] = -f.x;
    matrix[3] = 0.0f;

    matrix[4] = s.y;
    matrix[5] = u.y;
    matrix[6] = -f.y;
    matrix[7] = 0.0f;

    matrix[8] = s.z;
    matrix[9] = u.z;
    matrix[10] = -f.z;
    matrix[11] = 0.0f;

    matrix[12] = -dot(s, eye); // dot is from helper_math.h
    matrix[13] = -dot(u, eye); // dot is from helper_math.h
    matrix[14] = dot(f, eye); // dot is from helper_math.h
    matrix[15] = 1.0f;
}

// Simple perspective matrix calculation
void calculate_perspective(float* matrix, float fovy, float aspect, float near, float far) {
    float tanHalfFovy = tanf(fovy / 2.0f * (float)M_PI / 180.0f); // Use float cast for M_PI
    matrix[0] = 1.0f / (aspect * tanHalfFovy);
    matrix[1] = 0.0f;
    matrix[2] = 0.0f;
    matrix[3] = 0.0f;

    matrix[4] = 0.0f;
    matrix[5] = 1.0f / (tanHalfFovy);
    matrix[6] = 0.0f;
    matrix[7] = 0.0f;

    matrix[8] = 0.0f;
    matrix[9] = 0.0f;
    matrix[10] = -(far + near) / (far - near);
    matrix[11] = -1.0f;

    matrix[12] = 0.0f;
    matrix[13] = 0.0f;
    matrix[14] = -(2.0f * far * near) / (far - near);
    matrix[15] = 0.0f;
}

int main() {
    // --- GLFW/GLEW Initialization ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "TinyHair Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Capture mouse

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // --- Load Hair Data ---
    HairLoader hairLoader;
    std::string dataPath = "../data/strands00001.data";
    std::vector<float3> allVertices;
    std::vector<GLuint> allIndices;
    GLuint primitiveRestartIndex = std::numeric_limits<GLuint>::max();
    size_t totalVertexCount = 0;
    float3 minCoord = make_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    float3 maxCoord = make_float3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    try {
        std::cout << "Attempting to load hair data from: " << dataPath << std::endl;
        hairLoader.load(dataPath);
        std::cout << "Successfully loaded hair data." << std::endl;
        std::cout << "Number of strands loaded: " << hairLoader.strands.size() << std::endl;

        // Consolidate vertices and create indices for Primitive Restart
        GLuint currentVertexOffset = 0;
        for (const auto& strand : hairLoader.strands) {
            if (strand.size() < 2) continue; // Need at least 2 vertices for a line strip

            // Add vertices to the combined list
            allVertices.insert(allVertices.end(), strand.begin(), strand.end());

            // Add indices for this strand
            for (size_t i = 0; i < strand.size(); ++i) {
                allIndices.push_back(currentVertexOffset + i);
            }
            // Add restart index after each strand
            allIndices.push_back(primitiveRestartIndex);

            currentVertexOffset += strand.size();
            totalVertexCount += strand.size(); // Keep track for logging/debugging if needed

            // Calculate bounding box (moved inside loop for efficiency)
            for (const auto& vertex : strand) {
                 minCoord.x = fminf(minCoord.x, vertex.x);
                 minCoord.y = fminf(minCoord.y, vertex.y);
                 minCoord.z = fminf(minCoord.z, vertex.z);
                 maxCoord.x = fmaxf(maxCoord.x, vertex.x);
                 maxCoord.y = fmaxf(maxCoord.y, vertex.y);
                 maxCoord.z = fmaxf(maxCoord.z, vertex.z);
            }
        }
        // Remove the last unnecessary restart index
        if (!allIndices.empty()) {
            allIndices.pop_back();
        }

        std::cout << "Total vertices consolidated: " << allVertices.size() << std::endl;
        std::cout << "Total indices created (incl. restarts): " << allIndices.size() << std::endl;
        std::cout << "Hair data bounding box:" << std::endl;
        std::cout << "  Min: (" << minCoord.x << ", " << minCoord.y << ", " << minCoord.z << ")" << std::endl;
        std::cout << "  Max: (" << maxCoord.x << ", " << maxCoord.y << ", " << maxCoord.z << ")" << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading hair data: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // --- Load Shaders ---
    Shader hairShader("../tinyhair/shader/vertex.glsl", "../tinyhair/shader/fragment.glsl"); // Corrected path
    if (hairShader.ID == 0) { // Check if shader loading failed (ID set to 0 in constructor on error)
         std::cerr << "Failed to load shaders." << std::endl;
         // Cleanup hair data VBOs/VAOs if they were created before this point
         glfwDestroyWindow(window);
         glfwTerminate();
         return -1;
    }

    // --- Prepare Single OpenGL Buffers for All Hair Strands ---
    GLuint hairVAO, hairVBO, hairIBO;

    glGenVertexArrays(1, &hairVAO);
    glGenBuffers(1, &hairVBO);
    glGenBuffers(1, &hairIBO);

    glBindVertexArray(hairVAO);

    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, hairVBO);
    glBufferData(GL_ARRAY_BUFFER, allVertices.size() * sizeof(float3), allVertices.data(), GL_STATIC_DRAW); // Use GL_DYNAMIC_DRAW if CUDA will modify it

    // Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, hairIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, allIndices.size() * sizeof(GLuint), allIndices.data(), GL_STATIC_DRAW);

    // Set up vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
    glEnableVertexAttribArray(0);

    // Unbind VAO (good practice)
    glBindVertexArray(0);
    // Unbind buffers (optional, VAO remembers IBO binding)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // --- Render Loop ---
    glEnable(GL_DEPTH_TEST); // Enable depth testing if needed, though lines might not need it
    glLineWidth(1.0f); // Set line width explicitly

    // Enable Primitive Restart using the fixed index (GLuint max)
    glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);

    while (!glfwWindowShouldClose(window)) {
        // Input handling (basic camera movement)
        float cameraSpeed = 0.05f; // Adjust as necessary
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos = cameraPos + cameraSpeed * cameraFront; // Use operator+ and operator* from helper_math.h
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos = cameraPos - cameraSpeed * cameraFront; // Use operator- and operator*
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos = cameraPos - normalize(cross(cameraFront, cameraUp)) * cameraSpeed; // Use operator-, normalize, cross, operator*
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos = cameraPos + normalize(cross(cameraFront, cameraUp)) * cameraSpeed; // Use operator+, normalize, cross, operator*
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Rendering commands here
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        hairShader.use();

        // Set up view and projection matrices
        float viewMatrix[16];
        float projectionMatrix[16];
        calculate_lookat(viewMatrix, cameraPos, cameraPos + cameraFront, cameraUp); // Use operator+ from helper_math.h
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        calculate_perspective(projectionMatrix, cameraZoom, (float)width / (float)height, 0.1f, 100.0f);

        // Pass matrices to shader (ensure uniform names match shader)
        glUniformMatrix4fv(glGetUniformLocation(hairShader.ID, "view"), 1, GL_FALSE, viewMatrix);
        glUniformMatrix4fv(glGetUniformLocation(hairShader.ID, "projection"), 1, GL_FALSE, projectionMatrix);

        // Check for OpenGL errors after setting uniforms
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error after setting uniforms: " << err << std::endl;
        }

        // Draw all hair strands with a single draw call
        glBindVertexArray(hairVAO);
        glDrawElements(GL_LINE_STRIP, static_cast<GLsizei>(allIndices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // Check for OpenGL errors after drawing
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error after drawing: " << err << std::endl;
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Disable Primitive Restart
    glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);

    // --- Cleanup ---
    glDeleteVertexArrays(1, &hairVAO);
    glDeleteBuffers(1, &hairVBO);
    glDeleteBuffers(1, &hairIBO);
    // Shader object is cleaned up by its destructor when hairShader goes out of scope

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
