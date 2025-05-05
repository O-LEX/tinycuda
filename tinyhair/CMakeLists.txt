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

# Set CUDA standard if needed
set_target_properties(tinyhair_test PROPERTIES CUDA_STANDARD 11)

# Copy shader files to the build directory next to the executable
add_custom_command(TARGET tinyhair_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/shader $<TARGET_FILE_DIR:tinyhair_test>/shader
    COMMENT "Copying shaders to build directory"
)

# Note: Shader files (vertex.glsl, fragment.glsl) are loaded relative to the executable's working directory
# or via absolute paths. Ensure they are accessible at runtime.
# For simplicity, we assume they are in a 'shader' subdirectory relative to the executable.
# A more robust solution involves installing shaders or configuring paths.
