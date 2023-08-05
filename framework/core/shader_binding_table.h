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
 * @brief Extended buffer class to simplify ray tracing shader binding table usage
 */
class ShaderBindingTable {
public:
    /**
	 * @brief Creates a shader binding table
	 * @param device A valid Vulkan device
	 * @param handle_count Shader group handle count
	 * @param handle_size_aligned Aligned shader group handle size
	 * @param memory_usage The memory usage of the shader binding table
	 */
    ShaderBindingTable(Device &device,
                       uint32_t handle_count,
                       VkDeviceSize handle_size_aligned,
                       VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU);

    ~ShaderBindingTable();

    const VkStridedDeviceAddressRegionKHR *get_strided_device_address_region() const;

    uint8_t *get_data() const;

private:
    Device &device;

    VkStridedDeviceAddressRegionKHR strided_device_address_region{};

    uint64_t device_address{0};

    VkBuffer handle{VK_NULL_HANDLE};

    VmaAllocation allocation{VK_NULL_HANDLE};

    VkDeviceMemory memory{VK_NULL_HANDLE};

    uint8_t *mapped_data{nullptr};
};
}// namespace core
}// namespace vox