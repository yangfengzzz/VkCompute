//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "framework/common/utils.h"
#include "framework/common/vk_common.h"
#include "framework/core/instance.h"
#include "framework/rendering/render_context.h"
#include "framework/rendering/render_pipeline.h"
#include "framework/stats/stats.h"
#include "application.h"

namespace vox {
/**
 * @mainpage Overview of the framework
 *
 * @section initialization Initialization
 *
 * @subsection platform_init Platform initialization
 * The lifecycle of a Vulkan sample starts by instantiating the correct Platform
 * (e.g. WindowsPlatform) and then calling initialize() on it, which sets up
 * the windowing system and logging. Then it calls the parent Platform::initialize(),
 * which takes ownership of the active application. It's the platforms responsibility
 * to then call GraphicsApplication::prepare() to prepare the vulkan sample when it is ready.
 *
 * @subsection sample_init Sample initialization
 * The preparation step is divided in two steps, one in GraphicsApplication and the other in the
 * specific sample, such as SurfaceRotation.
 * GraphicsApplication::prepare() contains functions that do not require customization,
 * including creating a Vulkan instance, the surface and getting physical devices.
 * The prepare() function for the specific sample completes the initialization, including:
 * - setting enabled Stats
 * - creating the Device
 * - creating the Swapchain
 * - creating the RenderContext (or child class)
 * - preparing the RenderContext
 * - loading the sg::Scene
 * - creating the RenderPipeline with ShaderModule (s)
 * - creating the sg::Camera
 * - creating the Gui
 *
 * @section frame_rendering Frame rendering
 *
 * @subsection update Update function
 * Rendering happens in the update() function. Each sample can override it, e.g.
 * to recreate the Swapchain in SwapchainImages when required by user input.
 * Typically a sample will then call GraphicsApplication::update().
 *
 * @subsection rendering Rendering
 * A series of steps are performed, some of which can be customized (it will be
 * highlighted when that's the case):
 *
 * - calling sg::Script::update() for all sg::Script (s)
 * - beginning a frame in RenderContext (does the necessary waiting on fences and
 *   acquires an core::Image)
 * - requesting a CommandBuffer
 * - updating Stats and Gui
 * - getting an active RenderTarget constructed by the factory function of the RenderFrame
 * - setting up barriers for color and depth, note that these are only for the default RenderTarget
 * - calling GraphicsApplication::draw_swapchain_renderpass (see below)
 * - setting up a barrier for the Swapchain transition to present
 * - submitting the CommandBuffer and end the Frame (present)
 *
 * @subsection draw_swapchain Draw swapchain renderpass
 * The function starts and ends a RenderPass which includes setting up viewport, scissors,
 * blend state (etc.) and calling draw_scene.
 * Note that RenderPipeline::draw is not virtual in RenderPipeline, but internally it calls
 * Subpass::draw for each Subpass, which is virtual and can be customized.
 *
 * @section framework_classes Main framework classes
 *
 * - RenderContext
 * - RenderFrame
 * - RenderTarget
 * - RenderPipeline
 * - ShaderModule
 * - ResourceCache
 * - BufferPool
 * - Core classes: Classes in vkb::core wrap Vulkan objects for indexing and hashing.
 */

class GraphicsApplication : public Application {
public:
    GraphicsApplication() = default;

    virtual ~GraphicsApplication();

    /**
	 * @brief Additional sample initialization
	 */
    bool prepare(const ApplicationOptions &options) override;

    /**
	 * @brief Create the Vulkan device used by this sample
	 * @note Can be overridden to implement custom device creation
	 */
    virtual void create_device();

    /**
	 * @brief Create the Vulkan instance used by this sample
	 * @note Can be overridden to implement custom instance creation
	 */
    virtual void create_instance();

    /**
	 * @brief Main loop sample events
	 */
    void update(float delta_time) override;

    bool resize(uint32_t width, uint32_t height) override;

    void input_event(const InputEvent &input_event) override;

    void finish() override;

    VkSurfaceKHR get_surface();

    core::Device &get_device();

    inline bool has_render_context() const {
        return render_context != nullptr;
    }

    rendering::RenderContext &get_render_context();

    void set_render_pipeline(rendering::RenderPipeline &&render_pipeline);

    rendering::RenderPipeline &get_render_pipeline();

protected:
    /**
	 * @brief The Vulkan instance
	 */
    std::unique_ptr<core::Instance> instance{nullptr};

    /**
	 * @brief The Vulkan device
	 */
    std::unique_ptr<core::Device> device{nullptr};

    /**
	 * @brief Context used for rendering, it is responsible for managing the frames and their underlying images
	 */
    std::unique_ptr<rendering::RenderContext> render_context{nullptr};

    /**
	 * @brief Pipeline used for rendering, it should be set up by the concrete sample
	 */
    std::unique_ptr<rendering::RenderPipeline> render_pipeline{nullptr};

    std::unique_ptr<Stats> stats{nullptr};

    /**
	 * @brief Update counter values
	 * @param delta_time
	 */
    void update_stats(float delta_time);

    /**
	 * @brief Prepares the render target and draws to it, calling draw_renderpass
	 * @param command_buffer The command buffer to record the commands to
	 * @param render_target The render target that is being drawn to
	 */
    virtual void draw(core::CommandBuffer &command_buffer, rendering::RenderTarget &render_target);

    /**
	 * @brief Starts the render pass, executes the render pipeline, and then ends the render pass
	 * @param command_buffer The command buffer to record the commands to
	 * @param render_target The render target that is being drawn to
	 */
    virtual void draw_renderpass(core::CommandBuffer &command_buffer, rendering::RenderTarget &render_target);

    /**
	 * @brief Triggers the render pipeline, it can be overridden by samples to specialize their rendering logic
	 * @param command_buffer The command buffer to record the commands to
	 */
    virtual void render(core::CommandBuffer &command_buffer);

    /**
	 * @brief Get additional sample-specific instance layers.
	 *
	 * @return Vector of additional instance layers. Default is empty vector.
	 */
    virtual const std::vector<const char *> get_validation_layers();

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
	 * @brief Set the Vulkan API version to request at instance creation time
	 */
    void set_api_version(uint32_t requested_api_version);

    /**
	 * @brief Request features from the gpu based on what is supported
	 */
    virtual void request_gpu_features(core::PhysicalDevice &gpu);

    /**
	 * @brief Override this to customise the creation of the render_context
	 */
    virtual void create_render_context();

    /**
	 * @brief Override this to customise the creation of the swapchain and render_context
	 */
    virtual void prepare_render_context();

    /**
	 * @brief Resets the stats view max values for high demanding configs
	 *        Should be overridden by the samples since they
	 *        know which configuration is resource demanding
	 */
    virtual void reset_stats_view(){};

    /**
	 * @brief Samples should override this function to draw their interface
	 */
    virtual void draw_gui();

    /**
	 * @brief Updates the debug window, samples can override this to insert their own data elements
	 */
    virtual void update_debug_window();

    /**
	 * @brief Set viewport and scissor state in command buffer for a given extent
	 */
    static void set_viewport_and_scissor(core::CommandBuffer &command_buffer, const VkExtent2D &extent);

    static constexpr float STATS_VIEW_RESET_TIME{10.0f};// 10 seconds

    /**
	 * @brief The Vulkan surface
	 */
    VkSurfaceKHR surface{VK_NULL_HANDLE};

    /**
	 * @brief Sets whether or not the first graphics queue should have higher priority than other queues.
	 * Very specific feature which is used by async compute samples.
	 * Needs to be called before prepare().
	 * @param enable If true, present queue will have prio 1.0 and other queues have prio 0.5.
	 * Default state is false, where all queues have 0.5 priority.
	 */
    void set_high_priority_graphics_queue_enable(bool enable) {
        high_priority_graphics_queue = enable;
    }

    /**
	 * @brief A helper to create a render context
	 */
    void create_render_context(const std::vector<VkSurfaceFormatKHR> &surface_formats);

private:
    /** @brief Set of device extensions to be enabled for this example and whether they are optional (must be set in the derived constructor) */
    std::unordered_map<const char *, bool> device_extensions;

    /** @brief Set of instance extensions to be enabled for this example and whether they are optional (must be set in the derived constructor) */
    std::unordered_map<const char *, bool> instance_extensions;

    /** @brief The Vulkan API version to request for this sample at instance creation time */
    uint32_t api_version = VK_API_VERSION_1_0;

    /** @brief Whether or not we want a high priority graphics queue. */
    bool high_priority_graphics_queue{false};
};
}// namespace vox
