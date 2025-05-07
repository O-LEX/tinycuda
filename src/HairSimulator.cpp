#include <GL/glew.h>

#include "HairSimulator.h"
#include <vector>

HairSimulator::HairSimulator() {
    // Members are initialized by their default initializers (nullptr, 0, false)
}

HairSimulator::~HairSimulator() {
    releaseResources();
}

void HairSimulator::releaseResources() {
    if (m_GpuData.d_posX) { cudaFree(m_GpuData.d_posX); m_GpuData.d_posX = nullptr; }
    if (m_GpuData.d_posY) { cudaFree(m_GpuData.d_posY); m_GpuData.d_posY = nullptr; }
    if (m_GpuData.d_posZ) { cudaFree(m_GpuData.d_posZ); m_GpuData.d_posZ = nullptr; }
    if (m_GpuData.d_strand_indices) { cudaFree(m_GpuData.d_strand_indices); m_GpuData.d_strand_indices = nullptr; }
    if (m_GpuData.d_particle_indices_in_strand) { cudaFree(m_GpuData.d_particle_indices_in_strand); m_GpuData.d_particle_indices_in_strand = nullptr; }
    m_GpuData.num_total_particles = 0;

    unmapVbo();
    cudaGraphicsUnregisterResource(m_vboCudaResource);
    deleteVbo(); 
}

void HairSimulator::deleteVbo() {
    glDeleteBuffers(1, &m_vboId);
    m_vboId = 0;
}


bool HairSimulator::initialize(const std::vector<std::vector<float3>>& raw_strands) {
    releaseResources(); 

    for (const auto& strand : raw_strands) {
        m_GpuData.num_total_particles += strand.size();
    }

    std::vector<float> h_posX(m_GpuData.num_total_particles);
    std::vector<float> h_posY(m_GpuData.num_total_particles);
    std::vector<float> h_posZ(m_GpuData.num_total_particles);
    std::vector<int> h_strand_indices(m_GpuData.num_total_particles);
    std::vector<int> h_particle_indices_in_strand(m_GpuData.num_total_particles);

    int current_particle_idx = 0;
    for (int strand_idx = 0; strand_idx < raw_strands.size(); ++strand_idx) {
        const auto& strand = raw_strands[strand_idx];
        for (int particle_in_strand_idx = 0; particle_in_strand_idx < strand.size(); ++particle_in_strand_idx) {
            const auto& particle = strand[particle_in_strand_idx];
            h_posX[current_particle_idx] = particle.x;
            h_posY[current_particle_idx] = particle.y;
            h_posZ[current_particle_idx] = particle.z;
            h_strand_indices[current_particle_idx] = strand_idx;
            h_particle_indices_in_strand[current_particle_idx] = particle_in_strand_idx;
            current_particle_idx++;
        }
    }

    // Allocate GPU memory and copy from host (error checks removed)
    cudaMalloc(reinterpret_cast<void**>(&m_GpuData.d_posX), m_GpuData.num_total_particles * sizeof(float));
    cudaMemcpy(m_GpuData.d_posX, h_posX.data(), m_GpuData.num_total_particles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&m_GpuData.d_posY), m_GpuData.num_total_particles * sizeof(float));
    cudaMemcpy(m_GpuData.d_posY, h_posY.data(), m_GpuData.num_total_particles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&m_GpuData.d_posZ), m_GpuData.num_total_particles * sizeof(float));
    cudaMemcpy(m_GpuData.d_posZ, h_posZ.data(), m_GpuData.num_total_particles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&m_GpuData.d_strand_indices), m_GpuData.num_total_particles * sizeof(int));
    cudaMemcpy(m_GpuData.d_strand_indices, h_strand_indices.data(), m_GpuData.num_total_particles * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&m_GpuData.d_particle_indices_in_strand), m_GpuData.num_total_particles * sizeof(int));
    cudaMemcpy(m_GpuData.d_particle_indices_in_strand, h_particle_indices_in_strand.data(), m_GpuData.num_total_particles * sizeof(int), cudaMemcpyHostToDevice);

    size_t vboSize = m_GpuData.num_total_particles * 3 * sizeof(float);

    glGenBuffers(1, &m_vboId);
    glBindBuffer(GL_ARRAY_BUFFER, m_vboId);
    glBufferData(GL_ARRAY_BUFFER, vboSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&m_vboCudaResource, m_vboId, cudaGraphicsRegisterFlagsWriteDiscard);
    return true; 
}

float* HairSimulator::mapVbo() {
    cudaGraphicsMapResources(1, &m_vboCudaResource, 0); // Error check removed

    float* d_vbo_ptr = nullptr;
    size_t mapped_vbo_buffer_size;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_vbo_ptr), &mapped_vbo_buffer_size, m_vboCudaResource); // Error check removed
    return d_vbo_ptr;
}

void HairSimulator::unmapVbo() {
    cudaGraphicsUnmapResources(1, &m_vboCudaResource, 0); // Error check removed
}

const GpuData& HairSimulator::getGpuData() const {
    return m_GpuData;
}

unsigned int HairSimulator::getVboId() const {
    return m_vboId;
}
