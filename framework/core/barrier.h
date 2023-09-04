//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "buffer.h"
#include "image.h"
#include <thsvs_simpler_vulkan_synchronization.h>

namespace vox::core {
struct ImageBarrier {
    VkImage image;
    ThsvsAccessType prev_access;
    ThsvsAccessType next_access;
    VkImageAspectFlags aspect_mask;
    bool discard;
};

void record_image_barrier(Device &device, VkCommandBuffer cb, const ImageBarrier &barrier);

}// namespace vox::core