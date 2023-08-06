//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "physical_device.h"

namespace vox {
PhysicalDevice::PhysicalDevice(Instance &instance, VkPhysicalDevice physical_device) : instance{instance},
                                                                                       handle{physical_device} {
    vkGetPhysicalDeviceFeatures(physical_device, &features);
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    LOGI("Found GPU: {}", properties.deviceName)

    uint32_t queue_family_properties_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_properties_count, nullptr);
    queue_family_properties = std::vector<VkQueueFamilyProperties>(queue_family_properties_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_properties_count, queue_family_properties.data());
}

Instance &PhysicalDevice::get_instance() const {
    return instance;
}

VkBool32 PhysicalDevice::is_present_supported(VkSurfaceKHR surface, uint32_t queue_family_index) const {
    VkBool32 present_supported{VK_FALSE};

    if (surface != VK_NULL_HANDLE) {
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(handle, queue_family_index, surface, &present_supported));
    }

    return present_supported;
}

VkFormatProperties PhysicalDevice::get_format_properties(VkFormat format) const {
    VkFormatProperties format_properties;

    vkGetPhysicalDeviceFormatProperties(handle, format, &format_properties);

    return format_properties;
}

VkPhysicalDevice PhysicalDevice::get_handle() const {
    return handle;
}

const VkPhysicalDeviceFeatures &PhysicalDevice::get_features() const {
    return features;
}

const VkPhysicalDeviceProperties &PhysicalDevice::get_properties() const {
    return properties;
}

const VkPhysicalDeviceMemoryProperties &PhysicalDevice::get_memory_properties() const {
    return memory_properties;
}

const std::vector<VkQueueFamilyProperties> &PhysicalDevice::get_queue_family_properties() const {
    return queue_family_properties;
}

uint32_t PhysicalDevice::get_queue_family_performance_query_passes(
    const VkQueryPoolPerformanceCreateInfoKHR *perf_query_create_info) const {
    uint32_t passes_needed;
    vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(get_handle(), perf_query_create_info,
                                                            &passes_needed);
    return passes_needed;
}

void PhysicalDevice::enumerate_queue_family_performance_query_counters(
    uint32_t queue_family_index,
    uint32_t *count,
    VkPerformanceCounterKHR *counters,
    VkPerformanceCounterDescriptionKHR *descriptions) const {
    VK_CHECK(vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
        get_handle(), queue_family_index, count, counters, descriptions));
}

VkPhysicalDeviceFeatures PhysicalDevice::get_requested_features() const {
    return requested_features;
}

VkPhysicalDeviceFeatures &PhysicalDevice::get_mutable_requested_features() {
    return requested_features;
}

void *PhysicalDevice::get_extension_feature_chain() const {
    return last_requested_extension_feature;
}

}// namespace vox
