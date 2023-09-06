//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/command_buffer.h"
#include <thsvs_simpler_vulkan_synchronization.h>

namespace vox::core {
struct ImageBarrier {
    core::Image& image;
    ThsvsAccessType prev_access{};
    ThsvsAccessType next_access{};
    VkImageAspectFlags aspect_mask{};
    bool discard{false};
};

struct BufferBarrier {
    core::Buffer& buffer;
    ThsvsAccessType prev_access{};
    ThsvsAccessType next_access{};
};

struct AccessInfo {
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkImageLayout image_layout;
};

void record_image_barrier(core::CommandBuffer &cb, const ImageBarrier &barrier);

void record_buffer_barrier(core::CommandBuffer &cb, const BufferBarrier &barrier);

AccessInfo get_access_info(ThsvsAccessType access_type);

VkImageAspectFlags image_aspect_mask_from_format(VkFormat format);

VkImageAspectFlags image_aspect_mask_from_access_type_and_format(ThsvsAccessType access_type, VkFormat format);

}// namespace vox::core