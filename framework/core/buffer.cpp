//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/buffer.h"
#include "core/buffer_pool.h"
#include "core/device.h"

namespace vox::core {
Buffer::Buffer(Device const &device,
               BufferDesc desc,
               BufferPool *pool,
               VmaAllocationCreateFlags flags,
               const std::vector<uint32_t> &queue_family_indices)
    : VulkanResource{VK_NULL_HANDLE, &device},
      size{desc.size} {
#ifdef VK_USE_PLATFORM_METAL_EXT
    // Workaround for Mac (MoltenVK requires unmapping https://github.com/KhronosGroup/MoltenVK/issues/175)
    // Force cleares the flag VMA_ALLOCATION_CREATE_MAPPED_BIT
    flags &= ~VMA_ALLOCATION_CREATE_MAPPED_BIT;
#endif

    persistent = (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = desc.buffer_usage;
    buffer_info.size = size;
    if (queue_family_indices.size() >= 2) {
        buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        buffer_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_family_indices.size());
        buffer_info.pQueueFamilyIndices = queue_family_indices.data();
    }
    if (pool && pool->is_exported()) {
        VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
        externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        externalMemoryBufferInfo.handleTypes = get_default_mem_handle_type();
        buffer_info.pNext = &externalMemoryBufferInfo;
    }

    VmaAllocationCreateInfo memory_info{};
    memory_info.flags = flags;
    memory_info.usage = desc.memory_usage;

    VmaAllocationInfo allocation_info{};
    auto result = vmaCreateBuffer(device.get_memory_allocator(),
                                  &buffer_info, &memory_info,
                                  &handle, &allocation,
                                  &allocation_info);

    if (result != VK_SUCCESS) {
        throw VulkanException{result, "Cannot create Buffer"};
    }

    memory = allocation_info.deviceMemory;

    if (persistent) {
        mapped_data = static_cast<uint8_t *>(allocation_info.pMappedData);
    }
}

Buffer::Buffer(Device const &device,
               void *raw_buffer,
               VkDeviceSize size,
               VkBufferUsageFlags usage,
               VkMemoryPropertyFlags properties)
    : VulkanResource{VK_NULL_HANDLE, &device},
      size{size} {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
    externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    bufferInfo.pNext = &externalMemoryBufferInfo;

    if (vkCreateBuffer(device.get_handle(), &bufferInfo, nullptr, &handle) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device.get_handle(), handle, &memRequirements);

    VkImportMemoryFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    handleInfo.pNext = nullptr;
    handleInfo.fd = (int)(uintptr_t)raw_buffer;
    handleInfo.handleType = get_default_mem_handle_type();

    VkMemoryAllocateInfo memAllocation = {};
    memAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocation.pNext = (void *)&handleInfo;
    memAllocation.allocationSize = memRequirements.size;
    memAllocation.memoryTypeIndex = device.get_gpu().find_memory_type(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device.get_handle(), &memAllocation, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to import allocation!");
    }

    vkBindBufferMemory(device.get_handle(), handle, memory, 0);
}

Buffer::Buffer(Buffer &&other) noexcept : VulkanResource{other.handle, other.device},
                                          allocation{other.allocation},
                                          memory{other.memory},
                                          size{other.size},
                                          mapped_data{other.mapped_data},
                                          mapped{other.mapped} {
    // Reset other handles to avoid releasing on destruction
    other.allocation = VK_NULL_HANDLE;
    other.memory = VK_NULL_HANDLE;
    other.mapped_data = nullptr;
    other.mapped = false;
}

Buffer::~Buffer() {
    if (handle != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
        unmap();
        vmaDestroyBuffer(device->get_memory_allocator(), handle, allocation);
    }
}

const VkBuffer *Buffer::get() const {
    return &handle;
}

VmaAllocation Buffer::get_allocation() const {
    return allocation;
}

VkDeviceMemory Buffer::get_memory() const {
    return memory;
}

VkDeviceSize Buffer::get_size() const {
    return size;
}

uint8_t *Buffer::map() {
    if (!mapped && !mapped_data) {
        VK_CHECK(vmaMapMemory(device->get_memory_allocator(), allocation, reinterpret_cast<void **>(&mapped_data)));
        mapped = true;
    }
    return mapped_data;
}

void Buffer::unmap() {
    if (mapped) {
        vmaUnmapMemory(device->get_memory_allocator(), allocation);
        mapped_data = nullptr;
        mapped = false;
    }
}

void Buffer::flush() const {
    vmaFlushAllocation(device->get_memory_allocator(), allocation, 0, size);
}

void Buffer::update(const std::vector<uint8_t> &data, size_t offset) {
    update(data.data(), data.size(), offset);
}

uint64_t Buffer::get_device_address() {
    VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    buffer_device_address_info.buffer = handle;
    return vkGetBufferDeviceAddressKHR(device->get_handle(), &buffer_device_address_info);
}

void Buffer::update(void *data, size_t bytes_size, size_t offset) {
    update(reinterpret_cast<const uint8_t *>(data), bytes_size, offset);
}

void Buffer::update(const uint8_t *data, const size_t bytes_size, const size_t offset) {
    if (persistent) {
        std::copy(data, data + bytes_size, mapped_data + offset);
        flush();
    } else {
        map();
        std::copy(data, data + bytes_size, mapped_data + offset);
        flush();
        unmap();
    }
}

int Buffer::get_memory_handle(VkExternalMemoryHandleTypeFlagBits handleType) const {
    int fd = -1;

    VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
    vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    vkMemoryGetFdInfoKHR.pNext = nullptr;
    vkMemoryGetFdInfoKHR.memory = memory;
    vkMemoryGetFdInfoKHR.handleType = handleType;

    if (vkGetMemoryFdKHR(device->get_handle(), &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
        LOGE("Failed to retrieve handle for buffer!");
    }
    return fd;
}

}// namespace vox::core
