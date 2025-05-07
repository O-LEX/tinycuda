#include "HairLoader.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void HairLoader::load_data(const std::string& fn)
{
    strands.clear();
    std::ifstream ifs(fn, std::ios_base::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open hair data file: " + fn);
    }

    int num_strands_in_file;
    ifs.read(reinterpret_cast<char*>(&num_strands_in_file), sizeof(int));

    if (num_strands_in_file <= 0) { // Also check for non-positive
        throw std::runtime_error("Invalid number of strands in file: " + std::to_string(num_strands_in_file));
    }
    strands.resize(num_strands_in_file);

    for (auto& s : strands) {
        int num_particles_in_strand;
        ifs.read(reinterpret_cast<char*>(&num_particles_in_strand), sizeof(int));
        if (num_particles_in_strand <= 0) { // Also check for non-positive
            throw std::runtime_error("Invalid number of particles in a strand: " + std::to_string(num_particles_in_strand));
        }
        s.resize(num_particles_in_strand);
        ifs.read(reinterpret_cast<char*>(s.data()), sizeof(float3) * num_particles_in_strand);
        if (ifs.fail()) { // Check stream state after read
             throw std::runtime_error("Error reading particle data for a strand from file: " + fn);
        }
    }

    // Check for unexpected extra data
    char dummy_check;
    ifs.read(&dummy_check, 1);
    if (!ifs.eof()) { // If not EOF, there's extra data
        throw std::runtime_error("Unexpected extra data at the end of file: " + fn);
    }
    // The original check for ifs.eof() before reading 1 byte was problematic if the file ended perfectly.
    // This new check attempts to read one more byte, and if it's not EOF, then there's extra data.
    // If it IS eof after trying to read, it means the file was consumed as expected.

    clean_data();
}

void HairLoader::clean_data()
{
    // Remove strands with fewer than 2 particles (i.e., single points or empty)
    std::vector<std::vector<float3>> validStrands;
    for (const auto& strand : strands) // Use const auto&
    {
        if (strand.size() > 1)
        {
            validStrands.push_back(strand);
        }
    }
    strands = validStrands;
}

void HairLoader::load(const std::string& fn)
{
    size_t dot_pos = fn.find_last_of(".");
    std::string ext = "";
    if (dot_pos != std::string::npos) {
        ext = fn.substr(dot_pos); // Get extension including dot
    }

    if (ext == ".data") {
        load_data(fn);
    }
    else {
        throw std::runtime_error("Unsupported file extension: '" + ext + "'. Only '.data' is supported for file: " + fn);
    }
}