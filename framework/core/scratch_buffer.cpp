//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "scratch_buffer.h"

#include "device.h"

namespace vox::core {
ScratchBuffer::ScratchBuffer(Device &device, VkDeviceSize size) : device{device},
                                                                  size{size} {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.size = size;

    VmaAllocationCreateInfo memory_info{};
    memory_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VmaAllocationInfo allocation_info{};
    auto result = vmaCreateBuffer(device.get_memory_allocator(),
                                  &buffer_info, &memory_info,
                                  &handle, &allocation,
                                  &allocation_info);

    if (result != VK_SUCCESS) {
        throw VulkanException{result, "Could not create Scratchbuffer"};
    }

    memory = allocation_info.deviceMemory;

    VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    buffer_device_address_info.buffer = handle;
    device_address = vkGetBufferDeviceAddressKHR(device.get_handle(), &buffer_device_address_info);
}

ScratchBuffer::~ScratchBuffer() {
    if (handle != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
        vmaDestroyBuffer(device.get_memory_allocator(), handle, allocation);
    }
}

VkBuffer ScratchBuffer::get_handle() const {
    return handle;
}

uint64_t ScratchBuffer::get_device_address() const {
    return device_address;
}

}// namespace vox::core
