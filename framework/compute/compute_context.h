//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/frame_resource.h"

namespace vox::compute {
/**
 * @brief ComputeContext acts as a frame manager for the sample, with a lifetime that is the
 * same as that of the Application itself. It acts as a container for RenderFrame objects,
 * swapping between them (begin_frame, end_frame) and forwarding requests for Vulkan resources
 * to the active frame. Note that it's guaranteed that there is always an active frame.
 * More than one frame can be in-flight in the GPU, thus the need for per-frame resources.
 */
class ComputeContext final {
public:
    /**
	 * @brief Constructor
	 * @param device A valid device
	 */
    explicit ComputeContext(core::Device &device);

    ComputeContext(const ComputeContext &) = delete;

    ComputeContext(ComputeContext &&) = delete;

    virtual ~ComputeContext() = default;

    ComputeContext &operator=(const ComputeContext &) = delete;

    ComputeContext &operator=(ComputeContext &&) = delete;

    /**
	 * @brief Prepares the RenderFrames for rendering
	 * @param thread_count The number of threads in the application, necessary to allocate this many resource pools for each RenderFrame
	 * @param create_render_target_func A function delegate, used to create a RenderTarget
	 */
    void prepare(size_t thread_count = 1);

    /**
	 * @brief Prepares the next available frame for rendering
	 * @param reset_mode How to reset the command buffer
	 * @returns A valid command buffer to record commands to be submitted
	 * Also ensures that there is an active frame if there is no existing active frame already
	 */
    core::CommandBuffer &begin(core::CommandBuffer::ResetMode reset_mode = core::CommandBuffer::ResetMode::ResetPool);

    /**
	 * @brief Submits the command buffer to the right queue
	 * @param command_buffer A command buffer containing recorded commands
	 */
    void submit(core::CommandBuffer &command_buffer);

    /**
	 * @brief Submits multiple command buffers to the right queue
	 * @param command_buffers Command buffers containing recorded commands
	 */
    void submit(const std::vector<core::CommandBuffer *> &command_buffers);

    /**
	 * @brief begin_frame
	 */
    void begin_frame();

    VkSemaphore submit(const core::Queue &queue, const std::vector<core::CommandBuffer *> &command_buffers,
                       VkSemaphore wait_semaphore, VkPipelineStageFlags wait_pipeline_stage);

    /**
	 * @brief Submits a command buffer related to a frame to a queue
	 */
    void submit(const core::Queue &queue, const std::vector<core::CommandBuffer *> &command_buffers);

    /**
	 * @brief Waits a frame to finish its rendering
	 */
    virtual void wait_frame();

    void end_frame(VkSemaphore semaphore);

    /**
	 * @brief An error should be raised if the frame is not active.
	 *        A frame is active after @ref begin_frame has been called.
	 * @return The current active frame
	 */
    core::FrameResource &get_active_frame();

    VkSemaphore request_semaphore();
    VkSemaphore request_semaphore_with_ownership();
    void release_owned_semaphore(VkSemaphore semaphore);

    core::Device &get_device();

    /**
	 * @brief Returns the WSI acquire semaphore. Only to be used in very special circumstances.
	 * @return The WSI acquire semaphore.
	 */
    VkSemaphore consume_acquired_semaphore();

private:
    core::Device &device;

    /// If swapchain exists, then this will be a present supported queue, else a graphics queue
    const core::Queue &queue;

    std::unique_ptr<core::FrameResource> frames;

    VkSemaphore acquired_semaphore{};

    bool prepared{false};

    /// Whether a frame is active or not
    bool frame_active{false};

    size_t thread_count{1};
};

}// namespace vox::compute
