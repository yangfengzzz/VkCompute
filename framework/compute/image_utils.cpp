//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "image_utils.h"
#include "core/buffer.h"
#include "core/device.h"

namespace vox::compute {
void set_device_image_via_staging_buffer(
    core::Device &device, core::Image &device_image,
    VkExtent3D image_dimensions, VkImageLayout to_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_setter) {
    auto stage_buffer = core::Buffer(device, core::BufferDesc{buffer_size_in_bytes,
                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                              VMA_MEMORY_USAGE_CPU_ONLY});
    auto src_staging_ptr = stage_buffer.map();
    staging_buffer_setter(src_staging_ptr, buffer_size_in_bytes);
    stage_buffer.unmap();

    auto &queue = device.get_queue_by_flags(VK_QUEUE_TRANSFER_BIT, 0);
    auto &cmd = device.request_command_buffer();
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmd.image_memory_barrier(device_image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    {
        VkBufferImageCopy image_copy;
        image_copy.imageExtent = image_dimensions;
        cmd.copy_buffer_to_image(stage_buffer, device_image, {image_copy});
    }
    cmd.image_memory_barrier(device_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to_layout);
    cmd.end();
    queue.submit(cmd, device.request_fence());
    device.get_fence_pool().wait();

    device.get_fence_pool().reset();
    device.get_command_pool().reset_pool();
}

void get_device_image_via_staging_buffer(
    core::Device &device, core::Image &device_image,
    VkExtent3D image_dimensions, VkImageLayout from_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_getter) {
    auto stage_buffer = core::Buffer(device, core::BufferDesc{buffer_size_in_bytes,
                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                              VMA_MEMORY_USAGE_CPU_ONLY});

    auto &queue = device.get_queue_by_flags(VK_QUEUE_TRANSFER_BIT, 0);
    auto &cmd = device.request_command_buffer();
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmd.image_memory_barrier(device_image, from_layout,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    {
        VkBufferImageCopy image_copy;
        image_copy.imageExtent = image_dimensions;
        cmd.copy_image_to_buffer(device_image, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, stage_buffer, {image_copy});
    }
    cmd.end();
    queue.submit(cmd, device.request_fence());
    device.get_fence_pool().wait();

    device.get_fence_pool().reset();
    device.get_command_pool().reset_pool();

    auto src_staging_ptr = stage_buffer.map();
    staging_buffer_getter(src_staging_ptr, buffer_size_in_bytes);
    stage_buffer.unmap();
}

}// namespace vox::compute