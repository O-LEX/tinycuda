cmake_minimum_required(VERSION 3.10)
project(cudahair_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA processing for .cu files
set(CMAKE_CUDA_STANDARD 11) # Or your desired CUDA standard
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Define the executable target
add_executable(cudahair_test
    main.cpp
    dummy_kernel.cu # Add the .cu file
    ../src/HairDataManager.cpp
    ../src/HairLoader.cpp
)

# Include directories
target_include_directories(cudahair_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include # For HairDataManager.h, HairLoader.h, helper_math.h
    ${CMAKE_CURRENT_SOURCE_DIR} # For dummy_kernel.cuh
)

# Link libraries
target_link_libraries(cudahair_test PRIVATE
    CUDA::cudart
)

# Optional: Specify CUDA architecture if needed, e.g., for sm_75
# set_target_properties(cudahair_test PROPERTIES CUDA_ARCHITECTURES "75")
