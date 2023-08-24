//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/frame_resource.h"

namespace vox::compute {
enum class LatencyMeasureMode {
    // time spent from queue submit to returning from queue wait
    kSystemSubmit,
    // system_submit subtracted by time for void dispatch
    kSystemDispatch,
    // Timestamp difference measured on GPU
    kGpuTimestamp,
};

struct LatencyMeasure {
    LatencyMeasureMode mode;
    double overhead_seconds;
};

/**
 * @brief ComputeContext acts as a frame manager for the sample, with a lifetime that is the
 * same as that of the Application itself. It acts as a container for RenderFrame objects,
 * swapping between them (begin_frame, end_frame) and forwarding requests for Vulkan resources
 * to the active frame. Note that it's guaranteed that there is always an active frame.
 * More than one frame can be in-flight in the GPU, thus the need for per-frame resources.
 */
class ComputeContext final {
public:
    LatencyMeasure latency_measure;

    /**
	 * @brief Constructor
	 */
    explicit ComputeContext();

    ComputeContext(const ComputeContext &) = delete;

    ComputeContext(ComputeContext &&) = delete;

    virtual ~ComputeContext() = default;

    ComputeContext &operator=(const ComputeContext &) = delete;

    ComputeContext &operator=(ComputeContext &&) = delete;

    /**
	 * @brief Add a sample-specific device extension
	 * @param extension The extension name
	 * @param optional (Optional) Whether the extension is optional
	 */
    void add_device_extension(const char *extension, bool optional = false);

    /**
	 * @brief Add a sample-specific instance extension
	 * @param extension The extension name
	 * @param optional (Optional) Whether the extension is optional
	 */
    void add_instance_extension(const char *extension, bool optional = false);

    /**
	 * @brief Get sample-specific instance extensions.
	 *
	 * @return Map of instance extensions and whether or not they are optional. Default is empty map.
	 */
    const std::unordered_map<const char *, bool> get_instance_extensions();

    /**
	 * @brief Get sample-specific device extensions.
	 *
	 * @return Map of device extensions and whether or not they are optional. Default is empty map.
	 */
    const std::unordered_map<const char *, bool> get_device_extensions();

    /**
	 * @brief Set the Vulkan API version to request at instance creation time
	 */
    void set_api_version(uint32_t requested_api_version);

    /**
	 * @brief Get additional sample-specific instance layers.
	 *
	 * @return Vector of additional instance layers. Default is empty vector.
	 */
    virtual const std::vector<const char *> get_validation_layers();

    /**
	 * @brief Request features from the gpu based on what is supported
	 */
    virtual void request_gpu_features(core::PhysicalDevice &gpu);

public:
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

private:
    /**
	 * @brief The Vulkan instance
	 */
    std::unique_ptr<core::Instance> instance{nullptr};

    /**
	 * @brief The Vulkan device
	 */
    std::unique_ptr<core::Device> device{nullptr};

    /** @brief Set of device extensions to be enabled for this example and whether they are optional (must be set in the derived constructor) */
    std::unordered_map<const char *, bool> device_extensions;

    /** @brief Set of instance extensions to be enabled for this example and whether they are optional (must be set in the derived constructor) */
    std::unordered_map<const char *, bool> instance_extensions;

    /** @brief The Vulkan API version to request for this sample at instance creation time */
    uint32_t api_version = VK_API_VERSION_1_0;

private:
    std::unique_ptr<core::FrameResource> frames;

    bool prepared{false};

    /// Whether a frame is active or not
    bool frame_active{false};

    size_t thread_count{1};
};

}// namespace vox::compute
