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

namespace core {
/**
 * @brief A simplified buffer class for creating temporary device local scratch buffers, used in e.g. ray tracing
 */
class ScratchBuffer {
public:
    /**
	 * @brief Creates a scratch buffer using VMA with pre-defined usage flags
	 * @param device A valid Vulkan device
	 * @param size The size in bytes of the buffer
	 */
    ScratchBuffer(Device &device,
                  VkDeviceSize size);

    ~ScratchBuffer();

    VkBuffer get_handle() const;

    uint64_t get_device_address() const;

    /**
	 * @return The size of the buffer
	 */
    VkDeviceSize get_size() const {
        return size;
    }

private:
    Device &device;

    uint64_t device_address{0};

    VkBuffer handle{VK_NULL_HANDLE};

    VmaAllocation allocation{VK_NULL_HANDLE};

    VkDeviceMemory memory{VK_NULL_HANDLE};

    VkDeviceSize size{0};
};
}// namespace core
}// namespace vox