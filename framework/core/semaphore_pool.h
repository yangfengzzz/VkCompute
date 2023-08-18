//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"

namespace vox {
class Device;

class SemaphorePool {
public:
    explicit SemaphorePool(Device &device);

    SemaphorePool(const SemaphorePool &) = delete;

    SemaphorePool(SemaphorePool &&other) = delete;

    ~SemaphorePool();

    SemaphorePool &operator=(const SemaphorePool &) = delete;

    SemaphorePool &operator=(SemaphorePool &&) = delete;

    VkSemaphore request_semaphore();
    VkSemaphore request_semaphore_with_ownership();
    void release_owned_semaphore(VkSemaphore semaphore);

    void reset();

    [[nodiscard]] uint32_t get_active_semaphore_count() const;

private:
    Device &device;

    std::vector<VkSemaphore> semaphores;
    std::vector<VkSemaphore> released_semaphores;

    uint32_t active_semaphore_count{0};
};
}// namespace vox
