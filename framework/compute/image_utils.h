//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/image.h"
#include <span>

namespace vox::compute {
// Sets data for a |device_image| via a CPU staging buffer by invoking
// |staging_buffer_setter| on the pointer pointing to the start of the CPU
// staging buffer.
//
// |device_image| is expected to have VK_IMAGE_USAGE_TRANSFER_DST_BIT bit.
//
// This function will discard the existing content in the image and transition
// it into |to_layout|.
void set_device_image_via_staging_buffer(
    core::Device &device, core::Image *device_image,
    VkExtent3D image_dimensions, VkImageLayout to_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_setter);

// Get data from a |device_image| via a CPU staging buffer by invoking
// |staging_buffer_getter| on the pointer pointing to the start of the CPU
// staging buffer.
//
// |device_image| is expected to have VK_IMAGE_USAGE_TRANSFER_SRC_BIT bit.
//
// This function will transition the image from |from_layout| to
// VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
void get_device_image_via_staging_buffer(
    core::Device &device, core::Image &device_image,
    VkExtent3D image_dimensions, VkImageLayout from_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_getter);
}// namespace vox::compute