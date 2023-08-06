//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "shader_binding_table.h"

#include "device.h"

namespace vox {
ShaderBindingTable::ShaderBindingTable(Device &device,
                                       uint32_t handle_count,
                                       VkDeviceSize handle_size_aligned,
                                       VmaMemoryUsage memory_usage) : device{device} {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.size = handle_count * handle_size_aligned;

    VmaAllocationCreateInfo memory_info{};
    memory_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    memory_info.usage = memory_usage;

    VmaAllocationInfo allocation_info{};
    auto result = vmaCreateBuffer(device.get_memory_allocator(),
                                  &buffer_info, &memory_info,
                                  &handle, &allocation,
                                  &allocation_info);

    if (result != VK_SUCCESS) {
        throw VulkanException{result, "Could not create ShaderBindingTable"};
    }

    memory = allocation_info.deviceMemory;

    VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    buffer_device_address_info.buffer = handle;
    strided_device_address_region.deviceAddress = vkGetBufferDeviceAddressKHR(device.get_handle(), &buffer_device_address_info);
    strided_device_address_region.stride = handle_size_aligned;
    strided_device_address_region.size = handle_count * handle_size_aligned;

    mapped_data = static_cast<uint8_t *>(allocation_info.pMappedData);
}

ShaderBindingTable::~ShaderBindingTable() {
    if (handle != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
        vmaDestroyBuffer(device.get_memory_allocator(), handle, allocation);
    }
}

const VkStridedDeviceAddressRegionKHR *vox::ShaderBindingTable::get_strided_device_address_region() const {
    return &strided_device_address_region;
}
uint8_t *ShaderBindingTable::get_data() const {
    return mapped_data;
}

}// namespace vox
