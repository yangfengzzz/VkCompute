//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "compute/compute_resource.h"

namespace vox::compute {
ComputeResource::ComputeResource(core::PhysicalDevice &gpu, core::Device &device, size_t thread_count)
    : core::FrameResource(device, thread_count),
      gpu(gpu) {
}

core::CommandBuffer &ComputeResource::begin(core::CommandBuffer::ResetMode reset_mode) {
    const auto &queue = device.get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0);
    return request_command_buffer(queue, reset_mode);
}

void ComputeResource::submit(core::CommandBuffer &command_buffer) {
    submit({&command_buffer});
}

void ComputeResource::submit(const std::vector<core::CommandBuffer *> &command_buffers) {
    submit(device.get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0), command_buffers);
}

void ComputeResource::submit(const core::Queue &queue, const std::vector<core::CommandBuffer *> &command_buffers) {
    std::vector<VkCommandBuffer> cmd_buf_handles(command_buffers.size(), VK_NULL_HANDLE);
    std::transform(command_buffers.begin(), command_buffers.end(), cmd_buf_handles.begin(),
                   [](const core::CommandBuffer *cmd_buf) { return cmd_buf->get_handle(); });

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};

    submit_info.commandBufferCount = to_u32(cmd_buf_handles.size());
    submit_info.pCommandBuffers = cmd_buf_handles.data();

    VkFence fence = request_fence();

    queue.submit({submit_info}, fence);
}

}// namespace vox::compute