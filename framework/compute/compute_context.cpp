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

    auto gpu_count = instance->get_gpu_count();
    for (int i = 0; i < gpu_count; ++i) {
        create_device(instance->get_gpu_at(i));
    }
}

void ComputeContext::create_device(core::PhysicalDevice &gpu) {
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

    devices.emplace_back(std::make_unique<core::Device>(gpu, VK_NULL_HANDLE, std::move(debug_utils), get_device_extensions()));
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

std::unordered_map<const char *, bool> ComputeContext::get_instance_extensions() {
    return instance_extensions;
}

std::unordered_map<const char *, bool> ComputeContext::get_device_extensions() {
    return device_extensions;
}

void ComputeContext::request_gpu_features(core::PhysicalDevice &gpu) {
    // To be overridden by sample
}

uint32_t ComputeContext::get_device_count() {
    return devices.size();
}

ComputeResource ComputeContext::get_resource(uint32_t index, size_t thread_count) {
    return ComputeResource(instance->get_gpu_at(index), *devices[index], thread_count);
}

}// namespace vox::compute
