//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/vk_common.h"
#include "core/vulkan_resource.h"

namespace vox::core {
class Device;

class Semaphore : public VulkanResource<VkSemaphore, VK_OBJECT_TYPE_BUFFER, const Device> {
public:
    explicit Semaphore(Device &device, bool is_exported = false);

    Semaphore(const Semaphore &) = delete;

    Semaphore(Semaphore &&sampler) noexcept;

    ~Semaphore() override;

    Semaphore &operator=(const Semaphore &) = delete;

    Semaphore &operator=(Semaphore &&) = delete;

    int get_semaphore_handle(VkExternalSemaphoreHandleTypeFlagBits handleType);

    static VkExternalSemaphoreHandleTypeFlagBits get_default_semaphore_handle_type() {
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    }
};
}// namespace vox::core