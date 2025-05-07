#include "HairLoader.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector> // Include vector for clean_data

void HairLoader::load_data(const std::string& fn)
{
    strands.clear(); // Clear existing data before loading new data
    std::ifstream ifs(fn, std::ios_base::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("failed in hair::load_data, cannot open the file");
    }

    int num;
    ifs.read(reinterpret_cast<char*>(&num), sizeof(int));

    if (!num) {
        throw std::runtime_error("failed in hair::load_data, not strands included");
    }
    strands.resize(num);

    for (auto& s : strands) {
        ifs.read(reinterpret_cast<char*>(&num), sizeof(int));
        if (!num) {
            throw std::runtime_error("failed in hair::load_data, a strand is empty");
        }
        s.resize(num);
        ifs.read(reinterpret_cast<char*>(s.data()), sizeof(float3) * num);
    }

    if (ifs.eof()) {
        throw std::runtime_error("failed in hair::load_data, more data expected");
    }
    ifs.read(reinterpret_cast<char*>(&num), 1);
    if (!ifs.eof()) {
        throw std::runtime_error("failed in hair::load_data, unknown data not parsed");
    }

    clean_data();
}

void HairLoader::clean_data()
{
    // Takes only actual strands
    std::vector<std::vector<float3>> cleanStrands;
    for (int i = 0; i < strands.size(); i++)
    {
        if (strands[i].size() > 1)
        {
            cleanStrands.push_back(strands[i]);
        }
    }

    // Copy
    strands = cleanStrands;
}

void HairLoader::load(const std::string& fn)
{
    size_t dot = fn.find_last_of(".");
    std::string ext = "";
    if (dot != std::string::npos) {
        ext = fn.substr(dot, fn.size() - dot);
    }
    if (ext == ".data") {
        load_data(fn);
    }
    else {
        // Only .data is supported now
        throw std::runtime_error("failed in hair::load, invalid or unsupported file extension. Only '.data' is supported.");
    }
}