//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "compute/compute_context.h"

namespace vox::compute {

ComputeContext::ComputeContext() {
    LOGI("Initializing Vulkan sample");

    VkResult result = volkInitialize();
    if (result) {
        throw VulkanException(result, "Failed to initialize volk.");
    }

    std::unique_ptr<core::DebugUtils> debug_utils{};

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

    if (!instance) {
        instance = std::make_unique<core::Instance>("vulkan compute", get_instance_extensions(), get_validation_layers(), true, api_version);
    }

    auto &gpu = instance->get_first_gpu();
    // Request sample required GPU features
    request_gpu_features(gpu);

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

    if (!device) {
        device = std::make_unique<core::Device>(gpu, VK_NULL_HANDLE, std::move(debug_utils), get_device_extensions());
    }
}

void ComputeContext::add_instance_extension(const char *extension, bool optional) {
    instance_extensions[extension] = optional;
}

void ComputeContext::add_device_extension(const char *extension, bool optional) {
    device_extensions[extension] = optional;
}

void ComputeContext::set_api_version(uint32_t requested_api_version) {
    api_version = requested_api_version;
}

const std::vector<const char *> ComputeContext::get_validation_layers() {
    return {};
}

const std::unordered_map<const char *, bool> ComputeContext::get_instance_extensions() {
    return instance_extensions;
}

const std::unordered_map<const char *, bool> ComputeContext::get_device_extensions() {
    return device_extensions;
}

void ComputeContext::request_gpu_features(core::PhysicalDevice &gpu) {
    // To be overridden by sample
}

//-----------------------------------------------------------------------------------------------------------------------------------------------
void ComputeContext::prepare(size_t count) {
    device->wait_idle();

    frames = std::make_unique<core::FrameResource>(*device, count);

    this->thread_count = count;
    this->prepared = true;
}

core::CommandBuffer &ComputeContext::begin(core::CommandBuffer::ResetMode reset_mode) {
    assert(prepared && "ComputeContext not prepared for rendering, call prepare()");

    if (!frame_active) {
        begin_frame();
    }

    const auto &queue = device->get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0);
    return get_active_frame().request_command_buffer(queue, reset_mode);
}

void ComputeContext::submit(core::CommandBuffer &command_buffer) {
    submit({&command_buffer});
}

void ComputeContext::submit(const std::vector<core::CommandBuffer *> &command_buffers) {
    assert(frame_active && "ComputeContext is inactive, cannot submit command buffer. Please call begin()");

    VkSemaphore render_semaphore = VK_NULL_HANDLE;
    submit(device->get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0), command_buffers);
    end_frame(render_semaphore);
}

void ComputeContext::begin_frame() {
    assert(!frame_active && "Frame is still active, please call end_frame");

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
    frame_active = false;
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
    return *device;
}

}// namespace vox::compute
