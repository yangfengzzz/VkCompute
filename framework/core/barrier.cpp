//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "barrier.h"

#include "device.h"

#define THSVS_SIMPLER_VULKAN_SYNCHRONIZATION_IMPLEMENTATION
#include <thsvs_simpler_vulkan_synchronization.h>

namespace vox::core {
void record_image_barrier(Device &device, VkCommandBuffer cb, const ImageBarrier &barrier) {
    auto range = VkImageSubresourceRange{
        .aspectMask = barrier.aspect_mask,
        .baseMipLevel = 0,
        .levelCount = 0,
        .baseArrayLayer = 0,
        .layerCount = 0,
    };

    ThsvsImageBarrier imageBarrier{
        .prevAccessCount = 1,
        .pPrevAccesses = &barrier.prev_access,
        .nextAccessCount = 1,
        .pNextAccesses = &barrier.next_access,
        .prevLayout = THSVS_IMAGE_LAYOUT_OPTIMAL,
        .nextLayout = THSVS_IMAGE_LAYOUT_OPTIMAL,
        .discardContents = barrier.discard,
        .srcQueueFamilyIndex = device.get_queue_family_index(VK_QUEUE_GRAPHICS_BIT),
        .dstQueueFamilyIndex = device.get_queue_family_index(VK_QUEUE_GRAPHICS_BIT),
        .image = barrier.image,
        .subresourceRange = range,
    };
    thsvsCmdPipelineBarrier(cb, nullptr,
                            0, nullptr,
                            1, &imageBarrier);
}

}// namespace vox::core