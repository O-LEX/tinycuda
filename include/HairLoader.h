#pragma once

#include <cuda_runtime.h> // Include CUDA runtime header
#include <string>
#include <vector>

class Loader
{
public:
    void load_data(const std::string& fn);
    void clean_data(); // makes sure there are not single point strands!

    void load(const std::string& fn);

    std::vector<std::vector<float3>> strands;
};