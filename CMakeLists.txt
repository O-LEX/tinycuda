cmake_minimum_required(VERSION 3.28)
project(tinycuda LANGUAGES CXX CUDA)

add_executable(tinycuda
    src/main.cpp
    src/tiny.cu
    src/Shader.cpp
    src/HairLoader.cpp
)

find_package(CUDAToolkit REQUIRED)

set_target_properties(tinycuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # CUDA_ARCHITECTURES "89" 
)

target_include_directories(tinycuda PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

target_link_libraries(tinycuda PRIVATE 
    OpenGL::GL
    glfw
    GLEW::GLEW
)

add_subdirectory(tinyhair)
add_subdirectory(cudahair)