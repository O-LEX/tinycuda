cmake_minimum_required(VERSION 3.10)
project(tinyhair_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

# Define the executable target
# Use sources from parent directory
add_executable(tinyhair_test
    main.cpp
    ../src/HairLoader.cpp
    ../src/Shader.cpp
)

# Include directories
target_include_directories(tinyhair_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include # For HairLoader.h, Shader.h, helper_math.h
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${glfw3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(tinyhair_test PRIVATE
    CUDA::cudart
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    glfw
)
