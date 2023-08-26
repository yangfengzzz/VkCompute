//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "buffer_utils.h"
#include "core/device.h"

namespace vox::compute {
void set_device_buffer_via_staging_buffer(
    core::Device &device, core::Buffer &device_buffer,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_setter) {
    auto stage_buffer = core::Buffer(device, buffer_size_in_bytes,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VMA_MEMORY_USAGE_CPU_ONLY);
    auto src_staging_ptr = stage_buffer.map();
    staging_buffer_setter(src_staging_ptr, buffer_size_in_bytes);
    stage_buffer.unmap();

    auto &queue = device.get_queue_by_flags(VK_QUEUE_TRANSFER_BIT, 0);
    auto &cmd = device.request_command_buffer();
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmd.copy_buffer(stage_buffer, device_buffer, buffer_size_in_bytes);
    cmd.end();
    queue.submit(cmd, device.request_fence());
    device.get_fence_pool().wait();

    device.get_fence_pool().reset();
    device.get_command_pool().reset_pool();
}

void get_device_buffer_via_staging_buffer(
    core::Device &device, core::Buffer &device_buffer,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_getter) {
    auto stage_buffer = core::Buffer(device, buffer_size_in_bytes,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VMA_MEMORY_USAGE_CPU_ONLY);

    auto &queue = device.get_queue_by_flags(VK_QUEUE_TRANSFER_BIT, 0);
    auto &cmd = device.request_command_buffer();
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmd.copy_buffer(device_buffer, stage_buffer, buffer_size_in_bytes);
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