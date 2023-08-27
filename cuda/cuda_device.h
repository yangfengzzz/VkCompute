//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace vox::compute {
class CudaDevice {
public:
    CudaDevice(uint8_t *vkDeviceUUID, size_t UUID_SIZE);

    const cudaDeviceProp &get_prop() const;

private:
    int cuda_device = -1;
    cudaDeviceProp prop{};
};
}// namespace vox::compute