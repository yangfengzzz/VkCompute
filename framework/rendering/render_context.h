//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/command_buffer.h"
#include "core/command_pool.h"
#include "core/descriptor_set.h"
#include "core/descriptor_set_layout.h"
#include "core/pipeline.h"
#include "core/pipeline_layout.h"
#include "core/queue.h"
#include "core/render_pass.h"
#include "core/pipeline_state.h"
#include "core/resource_cache.h"
#include "rendering/render_frame.h"
#include "rendering/render_target.h"
#include "rendering/swapchain.h"
#include "rendering/framebuffer.h"
#include "shader/shader_module.h"

namespace vox {
class Window;

namespace rendering {

/**
 * @brief RenderContext acts as a frame manager for the sample, with a lifetime that is the
 * same as that of the Application itself. It acts as a container for RenderFrame objects,
 * swapping between them (begin_frame, end_frame) and forwarding requests for Vulkan resources
 * to the active frame. Note that it's guaranteed that there is always an active frame.
 * More than one frame can be in-flight in the GPU, thus the need for per-frame resources.
 *
 * It requires a Device to be valid on creation, and will take control of a given Swapchain.
 *
 * For normal rendering (using a swapchain), the RenderContext can be created by passing in a
 * swapchain. A RenderFrame will then be created for each Swapchain image.
 *
 * For headless rendering (no swapchain), the RenderContext can be given a valid Device, and
 * a width and height. A single RenderFrame will then be created.
 */
class RenderContext {
public:
    // The format to use for the RenderTargets if a swapchain isn't created
    static VkFormat DEFAULT_VK_FORMAT;

    /**
	 * @brief Constructor
	 * @param device A valid device
	 * @param surface A surface, VK_NULL_HANDLE if in headless mode
	 * @param window The window where the surface was created
	 * @param present_mode Requests to set the present mode of the swapchain
	 * @param present_mode_priority_list The order in which the swapchain prioritizes selecting its present mode
	 * @param surface_format_priority_list The order in which the swapchain prioritizes selecting its surface format
	 */
    RenderContext(core::Device &device,
                  VkSurfaceKHR surface,
                  const Window &window,
                  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR,
                  const std::vector<VkPresentModeKHR> &present_mode_priority_list = {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR},
                  const std::vector<VkSurfaceFormatKHR> &surface_format_priority_list = {
                      {VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
                      {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}});

    RenderContext(const RenderContext &) = delete;

    RenderContext(RenderContext &&) = delete;

    virtual ~RenderContext() = default;

    RenderContext &operator=(const RenderContext &) = delete;

    RenderContext &operator=(RenderContext &&) = delete;

    /**
	 * @brief Prepares the RenderFrames for rendering
	 * @param thread_count The number of threads in the application, necessary to allocate this many resource pools for each RenderFrame
	 * @param create_render_target_func A function delegate, used to create a RenderTarget
	 */
    void prepare(size_t thread_count = 1, RenderTarget::CreateFunc create_render_target_func = RenderTarget::DEFAULT_CREATE_FUNC);

    /**
	 * @brief Updates the swapchains extent, if a swapchain exists
	 * @param extent The width and height of the new swapchain images
	 */
    void update_swapchain(const VkExtent2D &extent);

    /**
	 * @brief Updates the swapchains image count, if a swapchain exists
	 * @param image_count The amount of images in the new swapchain
	 */
    void update_swapchain(const uint32_t image_count);

    /**
	 * @brief Updates the swapchains image usage, if a swapchain exists
	 * @param image_usage_flags The usage flags the new swapchain images will have
	 */
    void update_swapchain(const std::set<VkImageUsageFlagBits> &image_usage_flags);

    /**
	 * @brief Updates the swapchains extent and surface transform, if a swapchain exists
	 * @param extent The width and height of the new swapchain images
	 * @param transform The surface transform flags
	 */
    void update_swapchain(const VkExtent2D &extent, const VkSurfaceTransformFlagBitsKHR transform);

    /**
	 * @returns True if a valid swapchain exists in the RenderContext
	 */
    bool has_swapchain();

    /**
	 * @brief Recreates the RenderFrames, called after every update
	 */
    void recreate();

    /**
	 * @brief Recreates the swapchain
	 */
    void recreate_swapchain();

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
    RenderFrame &get_active_frame();

    /**
	 * @brief An error should be raised if the frame is not active.
	 *        A frame is active after @ref begin_frame has been called.
	 * @return The current active frame index
	 */
    uint32_t get_active_frame_index();

    /**
	 * @brief An error should be raised if a frame is active.
	 *        A frame is active after @ref begin_frame has been called.
	 * @return The previous frame
	 */
    RenderFrame &get_last_rendered_frame();

    VkSemaphore request_semaphore();
    VkSemaphore request_semaphore_with_ownership();
    void release_owned_semaphore(VkSemaphore semaphore);

    core::Device &get_device();

    /**
	 * @brief Returns the format that the RenderTargets are created with within the RenderContext
	 */
    VkFormat get_format() const;

    Swapchain const &get_swapchain() const;

    VkExtent2D const &get_surface_extent() const;

    uint32_t get_active_frame_index() const;

    std::vector<std::unique_ptr<RenderFrame>> &get_render_frames();

    /**
	 * @brief Handles surface changes, only applicable if the render_context makes use of a swapchain
	 */
    virtual bool handle_surface_changes(bool force_update = false);

    /**
	 * @brief Returns the WSI acquire semaphore. Only to be used in very special circumstances.
	 * @return The WSI acquire semaphore.
	 */
    VkSemaphore consume_acquired_semaphore();

protected:
    VkExtent2D surface_extent;

private:
    core::Device &device;

    const Window &window;

    /// If swapchain exists, then this will be a present supported queue, else a graphics queue
    const core::Queue &queue;

    std::unique_ptr<Swapchain> swapchain;

    SwapchainProperties swapchain_properties;

    std::vector<std::unique_ptr<RenderFrame>> frames;

    VkSemaphore acquired_semaphore;

    bool prepared{false};

    /// Current active frame index
    uint32_t active_frame_index{0};

    /// Whether a frame is active or not
    bool frame_active{false};

    RenderTarget::CreateFunc create_render_target_func = RenderTarget::DEFAULT_CREATE_FUNC;

    VkSurfaceTransformFlagBitsKHR pre_transform{VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR};

    size_t thread_count{1};
};

}// namespace rendering
}// namespace vox
