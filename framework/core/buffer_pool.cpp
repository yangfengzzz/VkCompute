//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/buffer_pool.h"
#include "core/device.h"

namespace vox::core {
BufferPool::BufferPool(core::Device &device, VkDeviceSize block_size, VkBufferUsageFlags usage,
                       VkExportMemoryAllocateInfoKHR *info)
    : device(device) {
    // Find memoryTypeIndex for the pool.
    VkBufferCreateInfo sampleBufCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sampleBufCreateInfo.size = 0x10000;// Doesn't matter.
    sampleBufCreateInfo.usage = usage;

    VmaAllocationCreateInfo sampleAllocCreateInfo = {};
    sampleAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    uint32_t memTypeIndex;
    VkResult result = vmaFindMemoryTypeIndexForBufferInfo(device.get_memory_allocator(),
                                                          &sampleBufCreateInfo,
                                                          &sampleAllocCreateInfo,
                                                          &memTypeIndex);
    if (result != VK_SUCCESS) {
        throw VulkanException{result, "Cannot find memory table index for buffer"};
    }

    // Create a pool that can have at most 2 blocks, 128 MiB each.
    VmaPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.memoryTypeIndex = memTypeIndex;
    poolCreateInfo.blockSize = block_size;
    poolCreateInfo.maxBlockCount = 2;
    poolCreateInfo.pMemoryAllocateNext = info;

    result = vmaCreatePool(device.get_memory_allocator(), &poolCreateInfo, &pool);
    if (result != VK_SUCCESS) {
        throw VulkanException{result, "Cannot create Buffer Pool"};
    }
}

BufferPool::~BufferPool() {
    VmaDetailedStatistics stats;
    vmaCalculatePoolStatistics(device.get_memory_allocator(), pool, &stats);
    auto allocated_bytes = stats.statistics.allocationBytes;
    if (allocated_bytes == 0) {
        vmaDestroyPool(device.get_memory_allocator(), pool);
        LOGI("Total device memory leaked: {} bytes.", allocated_bytes)
    } else {
        LOGE("Total device memory leaked: {} bytes.", allocated_bytes)
    };
}

}// namespace vox::core