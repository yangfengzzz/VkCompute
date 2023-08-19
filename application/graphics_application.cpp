//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "graphics_application.h"

#include "framework/platform/platform.h"

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#include "framework/platform/android/android_platform.h"
#endif

namespace vox {
GraphicsApplication::~GraphicsApplication() {
    if (device) {
        device->wait_idle();
    }

    stats.reset();
    render_context.reset();
    device.reset();

    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance->get_handle(), surface, nullptr);
    }

    instance.reset();
}

void GraphicsApplication::set_render_pipeline(rendering::RenderPipeline &&rp) {
    render_pipeline = std::make_unique<rendering::RenderPipeline>(std::move(rp));
}

rendering::RenderPipeline &GraphicsApplication::get_render_pipeline() {
    assert(render_pipeline && "Render pipeline was not created");
    return *render_pipeline;
}

bool GraphicsApplication::prepare(const ApplicationOptions &options) {
    if (!Application::prepare(options)) {
        return false;
    }

    LOGI("Initializing Vulkan sample");

    bool headless = window->get_window_mode() == Window::Mode::Headless;

    VkResult result = volkInitialize();
    if (result) {
        throw VulkanException(result, "Failed to initialize volk.");
    }

    std::unique_ptr<core::DebugUtils> debug_utils{};

    // Creating the vulkan instance
    for (const char *extension_name : window->get_required_surface_extensions()) {
        add_instance_extension(extension_name);
    }

#ifdef VKB_VULKAN_DEBUG
    {
        uint32_t instance_extension_count;
        VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr));

        std::vector<VkExtensionProperties> available_instance_extensions(instance_extension_count);
        VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, available_instance_extensions.data()));

        for (const auto &it : available_instance_extensions) {
            if (strcmp(it.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                LOGI("Vulkan debug utils enabled ({})", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

                debug_utils = std::make_unique<core::DebugUtilsExtDebugUtils>();
                add_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                break;
            }
        }
    }
#endif

    create_instance();

    if (!instance) {
        instance = std::make_unique<core::Instance>(get_name(), get_instance_extensions(), get_validation_layers(), headless, api_version);
    }

    // Getting a valid vulkan surface from the platform
    surface = window->create_surface(*instance);
    if (!surface) {
        throw std::runtime_error("Failed to create window surface.");
    }

    auto &gpu = instance->get_suitable_gpu(surface);
    gpu.set_high_priority_graphics_queue_enable(high_priority_graphics_queue);

    // Request to enable ASTC
    if (gpu.get_features().textureCompressionASTC_LDR) {
        gpu.get_mutable_requested_features().textureCompressionASTC_LDR = VK_TRUE;
    }

    // Request sample required GPU features
    request_gpu_features(gpu);

    // Creating vulkan device, specifying the swapchain extension always
    if (!headless || instance->is_enabled(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)) {
        add_device_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

        if (instance_extensions.find(VK_KHR_DISPLAY_EXTENSION_NAME) != instance_extensions.end()) {
            add_device_extension(VK_KHR_DISPLAY_SWAPCHAIN_EXTENSION_NAME, /*optional=*/true);
        }
    }

#ifdef VKB_VULKAN_DEBUG
    if (!debug_utils) {
        uint32_t device_extension_count;
        VK_CHECK(vkEnumerateDeviceExtensionProperties(gpu.get_handle(), nullptr, &device_extension_count, nullptr));

        std::vector<VkExtensionProperties> available_device_extensions(device_extension_count);
        VK_CHECK(vkEnumerateDeviceExtensionProperties(gpu.get_handle(), nullptr, &device_extension_count, available_device_extensions.data()));

        for (const auto &it : available_device_extensions) {
            if (strcmp(it.extensionName, VK_EXT_DEBUG_MARKER_EXTENSION_NAME) == 0) {
                LOGI("Vulkan debug utils enabled ({})", VK_EXT_DEBUG_MARKER_EXTENSION_NAME);

                debug_utils = std::make_unique<core::DebugMarkerExtDebugUtils>();
                add_device_extension(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
                break;
            }
        }
    }

    if (!debug_utils) {
        LOGW("Vulkan debug utils were requested, but no extension that provides them was found");
    }
#endif

    if (!debug_utils) {
        debug_utils = std::make_unique<core::DummyDebugUtils>();
    }

    create_device();// create_custom_device? better way than override?

    if (!device) {
        device = std::make_unique<core::Device>(gpu, surface, std::move(debug_utils), get_device_extensions());
    }

    create_render_context();
    prepare_render_context();

    stats = std::make_unique<Stats>(*render_context);

    return true;
}

void GraphicsApplication::create_device() {
}

void GraphicsApplication::create_instance() {
}

void GraphicsApplication::create_render_context() {
    auto surface_priority_list = std::vector<VkSurfaceFormatKHR>{{VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}, {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
    create_render_context(surface_priority_list);
}

void GraphicsApplication::create_render_context(const std::vector<VkSurfaceFormatKHR> &surface_priority_list) {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    VkPresentModeKHR present_mode = (window->get_properties().vsync == Window::Vsync::OFF) ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
    std::vector<VkPresentModeKHR> present_mode_priority_list{VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
#else
    VkPresentModeKHR present_mode = (window->get_properties().vsync == Window::Vsync::ON) ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR;
    std::vector<VkPresentModeKHR> present_mode_priority_list{VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
#endif

    render_context = std::make_unique<rendering::RenderContext>(get_device(), surface, *window, present_mode, present_mode_priority_list, surface_priority_list);
}

void GraphicsApplication::prepare_render_context() {
    render_context->prepare();
}

void GraphicsApplication::update_stats(float delta_time) {
    if (stats) {
        stats->update(delta_time);

        static float stats_view_count = 0.0f;
        stats_view_count += delta_time;

        // Reset every STATS_VIEW_RESET_TIME seconds
        if (stats_view_count > STATS_VIEW_RESET_TIME) {
            reset_stats_view();
            stats_view_count = 0.0f;
        }
    }
}

void GraphicsApplication::update(float delta_time) {
    auto &command_buffer = render_context->begin();

    // Collect the performance data for the sample graphs
    update_stats(delta_time);

    command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    stats->begin_sampling(command_buffer);

    draw(command_buffer, render_context->get_active_frame().get_render_target());

    stats->end_sampling(command_buffer);
    command_buffer.end();

    render_context->submit(command_buffer);
}

void GraphicsApplication::draw(core::CommandBuffer &command_buffer, rendering::RenderTarget &render_target) {
    auto &views = render_target.get_views();
    assert(1 < views.size());

    {
        // Image 0 is the swapchain
        ImageMemoryBarrier memory_barrier{};
        memory_barrier.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        memory_barrier.new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        memory_barrier.src_access_mask = 0;
        memory_barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        memory_barrier.src_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        memory_barrier.dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        command_buffer.image_memory_barrier(views[0], memory_barrier);

        // Skip 1 as it is handled later as a depth-stencil attachment
        for (size_t i = 2; i < views.size(); ++i) {
            command_buffer.image_memory_barrier(views[i], memory_barrier);
        }
    }

    {
        ImageMemoryBarrier memory_barrier{};
        memory_barrier.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        memory_barrier.new_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        memory_barrier.src_access_mask = 0;
        memory_barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        memory_barrier.src_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        memory_barrier.dst_stage_mask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

        command_buffer.image_memory_barrier(views[1], memory_barrier);
    }

    draw_renderpass(command_buffer, render_target);

    {
        ImageMemoryBarrier memory_barrier{};
        memory_barrier.old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        memory_barrier.new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        memory_barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        memory_barrier.src_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        memory_barrier.dst_stage_mask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

        command_buffer.image_memory_barrier(views[0], memory_barrier);
    }
}

void GraphicsApplication::draw_renderpass(core::CommandBuffer &command_buffer, rendering::RenderTarget &render_target) {
    set_viewport_and_scissor(command_buffer, render_target.get_extent());
    render(command_buffer);
    command_buffer.end_render_pass();
}

void GraphicsApplication::render(core::CommandBuffer &command_buffer) {
    if (render_pipeline) {
        render_pipeline->draw(command_buffer, render_context->get_active_frame().get_render_target());
    }
}

bool GraphicsApplication::resize(uint32_t width, uint32_t height) {
    Application::resize(width, height);

    if (stats) {
        stats->resize(width);
    }
    return true;
}

void GraphicsApplication::input_event(const InputEvent &input_event) {
    Application::input_event(input_event);
}

void GraphicsApplication::finish() {
    Application::finish();

    if (device) {
        device->wait_idle();
    }
}

core::Device &GraphicsApplication::get_device() {
    return *device;
}

void GraphicsApplication::draw_gui() {
}

void GraphicsApplication::update_debug_window() {
    auto driver_version = device->get_driver_version();
    std::string driver_version_str = fmt::format("major: {} minor: {} patch: {}", driver_version.major, driver_version.minor, driver_version.patch);
    get_debug_info().insert<field::Static, std::string>("driver_version", driver_version_str);

    get_debug_info().insert<field::Static, std::string>("resolution",
                                                        to_string(render_context->get_swapchain().get_extent()));

    get_debug_info().insert<field::Static, std::string>("surface_format",
                                                        to_string(render_context->get_swapchain().get_format()) + " (" +
                                                            to_string(get_bits_per_pixel(render_context->get_swapchain().get_format())) + "bpp)");
}

void GraphicsApplication::set_viewport_and_scissor(core::CommandBuffer &command_buffer, const VkExtent2D &extent) {
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    command_buffer.set_viewport(0, {viewport});

    VkRect2D scissor{};
    scissor.extent = extent;
    command_buffer.set_scissor(0, {scissor});
}

VkSurfaceKHR GraphicsApplication::get_surface() {
    return surface;
}

rendering::RenderContext &GraphicsApplication::get_render_context() {
    assert(render_context && "Render context is not valid");
    return *render_context;
}

const std::vector<const char *> GraphicsApplication::get_validation_layers() {
    return {};
}

const std::unordered_map<const char *, bool> GraphicsApplication::get_instance_extensions() {
    return instance_extensions;
}

const std::unordered_map<const char *, bool> GraphicsApplication::get_device_extensions() {
    return device_extensions;
}

void GraphicsApplication::add_device_extension(const char *extension, bool optional) {
    device_extensions[extension] = optional;
}

void GraphicsApplication::add_instance_extension(const char *extension, bool optional) {
    instance_extensions[extension] = optional;
}

void GraphicsApplication::set_api_version(uint32_t requested_api_version) {
    api_version = requested_api_version;
}

void GraphicsApplication::request_gpu_features(core::PhysicalDevice &gpu) {
    // To be overridden by sample
}

}// namespace vox
