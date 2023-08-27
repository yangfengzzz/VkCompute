//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include <cstdint>
#include "linmath.h"

class SineWaveSimulation {
    float *m_heightMap;
    size_t m_width, m_height;
    int m_blocks, m_threads;

public:
    SineWaveSimulation(size_t width, size_t height);
    ~SineWaveSimulation();
    void initSimulation(float *heightMap);
    void stepSimulation(float time, cudaStream_t stream = nullptr);
    void initCudaLaunchConfig(int device);
    int initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE);

    [[nodiscard]] size_t getWidth() const { return m_width; }
    [[nodiscard]] size_t getHeight() const { return m_height; }
};