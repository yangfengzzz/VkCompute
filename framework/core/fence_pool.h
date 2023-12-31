//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"

namespace vox::core {
class Device;

class FencePool {
public:
    explicit FencePool(Device &device);

    FencePool(const FencePool &) = delete;

    FencePool(FencePool &&other) = delete;

    ~FencePool();

    FencePool &operator=(const FencePool &) = delete;

    FencePool &operator=(FencePool &&) = delete;

    VkFence request_fence();

    VkResult wait(uint32_t timeout = std::numeric_limits<uint32_t>::max()) const;

    VkResult reset();

private:
    Device &device;

    std::vector<VkFence> fences;

    uint32_t active_fence_count{0};
};

}// namespace vox::core
