//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "compute/compute_context.h"

namespace vox::compute {

ComputeContext::ComputeContext(core::Device &device)
    : device{device},
      queue{device.get_suitable_graphics_queue()} {
}

void ComputeContext::prepare(size_t thread_count) {
    device.wait_idle();

    frames = std::make_unique<core::FrameResource>(device, thread_count);

    this->thread_count = thread_count;
    this->prepared = true;
}

core::CommandBuffer &ComputeContext::begin(core::CommandBuffer::ResetMode reset_mode) {
    assert(prepared && "ComputeContext not prepared for rendering, call prepare()");

    if (!frame_active) {
        begin_frame();
    }

    if (acquired_semaphore == VK_NULL_HANDLE) {
        throw std::runtime_error("Couldn't begin frame");
    }

    const auto &queue = device.get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0);
    return get_active_frame().request_command_buffer(queue, reset_mode);
}

void ComputeContext::submit(core::CommandBuffer &command_buffer) {
    submit({&command_buffer});
}

void ComputeContext::submit(const std::vector<core::CommandBuffer *> &command_buffers) {
    assert(frame_active && "ComputeContext is inactive, cannot submit command buffer. Please call begin()");

    VkSemaphore render_semaphore = VK_NULL_HANDLE;
    submit(queue, command_buffers);
    end_frame(render_semaphore);
}

void ComputeContext::begin_frame() {
    assert(!frame_active && "Frame is still active, please call end_frame");

    // We will use the acquired semaphore in a different frame context,
    // so we need to hold ownership.
    acquired_semaphore = frames->request_semaphore_with_ownership();

    // Now the frame is active again
    frame_active = true;

    // Wait on all resource to be freed from the previous render to this frame
    wait_frame();
}

VkSemaphore ComputeContext::submit(const core::Queue &queue, const std::vector<core::CommandBuffer *> &command_buffers,
                                   VkSemaphore wait_semaphore, VkPipelineStageFlags wait_pipeline_stage) {
    std::vector<VkCommandBuffer> cmd_buf_handles(command_buffers.size(), VK_NULL_HANDLE);
    std::transform(command_buffers.begin(), command_buffers.end(), cmd_buf_handles.begin(), [](const core::CommandBuffer *cmd_buf) { return cmd_buf->get_handle(); });

    auto &frame = get_active_frame();

    VkSemaphore signal_semaphore = frame.request_semaphore();

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};

    submit_info.commandBufferCount = to_u32(cmd_buf_handles.size());
    submit_info.pCommandBuffers = cmd_buf_handles.data();

    if (wait_semaphore != VK_NULL_HANDLE) {
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &wait_semaphore;
        submit_info.pWaitDstStageMask = &wait_pipeline_stage;
    }

    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &signal_semaphore;

    VkFence fence = frame.request_fence();

    queue.submit({submit_info}, fence);

    return signal_semaphore;
}

void ComputeContext::submit(const core::Queue &queue, const std::vector<core::CommandBuffer *> &command_buffers) {
    std::vector<VkCommandBuffer> cmd_buf_handles(command_buffers.size(), VK_NULL_HANDLE);
    std::transform(command_buffers.begin(), command_buffers.end(), cmd_buf_handles.begin(), [](const core::CommandBuffer *cmd_buf) { return cmd_buf->get_handle(); });

    auto &frame = get_active_frame();

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};

    submit_info.commandBufferCount = to_u32(cmd_buf_handles.size());
    submit_info.pCommandBuffers = cmd_buf_handles.data();

    VkFence fence = frame.request_fence();

    queue.submit({submit_info}, fence);
}

void ComputeContext::wait_frame() {
    auto &frame = get_active_frame();
    frame.reset();
}

void ComputeContext::end_frame(VkSemaphore semaphore) {
    assert(frame_active && "Frame is not active, please call begin_frame");

    // Frame is not active anymore
    if (acquired_semaphore) {
        release_owned_semaphore(acquired_semaphore);
        acquired_semaphore = VK_NULL_HANDLE;
    }
    frame_active = false;
}

VkSemaphore ComputeContext::consume_acquired_semaphore() {
    assert(frame_active && "Frame is not active, please call begin_frame");
    auto sem = acquired_semaphore;
    acquired_semaphore = VK_NULL_HANDLE;
    return sem;
}

core::FrameResource &ComputeContext::get_active_frame() {
    assert(frame_active && "Frame is not active, please call begin_frame");
    return *frames;
}

VkSemaphore ComputeContext::request_semaphore() {
    auto &frame = get_active_frame();
    return frame.request_semaphore();
}

VkSemaphore ComputeContext::request_semaphore_with_ownership() {
    auto &frame = get_active_frame();
    return frame.request_semaphore_with_ownership();
}

void ComputeContext::release_owned_semaphore(VkSemaphore semaphore) {
    auto &frame = get_active_frame();
    frame.release_owned_semaphore(semaphore);
}

core::Device &ComputeContext::get_device() {
    return device;
}

}// namespace vox::compute
