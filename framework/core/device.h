//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/logging.h"
#include "common/vk_common.h"
#include "core/command_buffer.h"
#include "core/command_pool.h"
#include "core/debug.h"
#include "core/descriptor_set.h"
#include "core/descriptor_set_layout.h"
#include "core/instance.h"
#include "core/physical_device.h"
#include "core/pipeline.h"
#include "core/pipeline_layout.h"
#include "core/queue.h"
#include "core/render_pass.h"
#include "shader/shader_module.h"
#include "core/vulkan_resource.h"
#include "core/fence_pool.h"
#include "core/pipeline_state.h"
#include "core/resource_cache.h"

#include "rendering/render_target.h"
#include "rendering/framebuffer.h"
#include "rendering/swapchain.h"

namespace vox::core {
struct DriverVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
};

class Device : public VulkanResource<VkDevice, VK_OBJECT_TYPE_DEVICE> {
public:
    /**
	 * @brief Device constructor
	 * @param gpu A valid Vulkan physical device and the requested gpu features
	 * @param surface The surface
	 * @param debug_utils The debug utils to be associated to this device
	 * @param requested_extensions (Optional) List of required device extensions and whether support is optional or not
	 */
    Device(PhysicalDevice &gpu,
           VkSurfaceKHR surface,
           std::unique_ptr<DebugUtils> &&debug_utils,
           std::unordered_map<const char *, bool> requested_extensions = {});

    /**
	 * @brief Device constructor
	 * @param gpu A valid Vulkan physical device and the requested gpu features
	 * @param vulkan_device A valid Vulkan device
	 * @param surface The surface
	 */
    Device(PhysicalDevice &gpu,
           VkDevice &vulkan_device,
           VkSurfaceKHR surface);

    Device(const Device &) = delete;

    Device(Device &&) = delete;

    ~Device() override;

    Device &operator=(const Device &) = delete;

    Device &operator=(Device &&) = delete;

    const PhysicalDevice &get_gpu() const;

    VmaAllocator get_memory_allocator() const;

    /**
	 * @brief Returns the debug utils associated with this Device.
	 */
    inline const DebugUtils &get_debug_utils() const {
        return *debug_utils;
    }

    /**
	 * @return The version of the driver of the current physical device
	 */
    DriverVersion get_driver_version() const;

    /**
	 * @return Whether an image format is supported by the GPU
	 */
    bool is_image_format_supported(VkFormat format) const;

    const Queue &get_queue(uint32_t queue_family_index, uint32_t queue_index);

    const Queue &get_queue_by_flags(VkQueueFlags queue_flags, uint32_t queue_index) const;

    const Queue &get_queue_by_present(uint32_t queue_index) const;

    /**
	 * @brief Manually adds a new queue from a given family index to this device
	 * @param global_index Index at where the queue should be placed inside the already existing list of queues
	 * @param family_index Index of the queue family from which the queue will be created
	 * @param properties Vulkan queue family properties
	 * @param can_present True if the queue is able to present images
	 */
    void add_queue(size_t global_index, uint32_t family_index, VkQueueFamilyProperties properties, VkBool32 can_present);

    /**
	 * @brief Finds a suitable graphics queue to submit to
	 * @return The first present supported queue, otherwise just any graphics queue
	 */
    const Queue &get_suitable_graphics_queue() const;

    bool is_extension_supported(const std::string &extension);

    bool is_enabled(const char *extension);

    uint32_t get_queue_family_index(VkQueueFlagBits queue_flag);

    uint32_t get_num_queues_for_queue_family(uint32_t queue_family_index);

    CommandPool &get_command_pool() const;

    /**
	 * @brief Checks that a given memory type is supported by the GPU
	 * @param bits The memory requirement type bits
	 * @param properties The memory property to search for
	 * @param memory_type_found True if found, false if not found
	 * @returns The memory type index of the found memory type
	 */
    uint32_t get_memory_type(uint32_t bits, VkMemoryPropertyFlags properties, VkBool32 *memory_type_found = nullptr) const;

    /**
	 * @brief Requests a command buffer from the general command_pool
	 * @return A new command buffer
	 */
    CommandBuffer &request_command_buffer() const;

    FencePool &get_fence_pool() const;

    /**
	 * @brief Creates the fence pool used by this device
	 */
    void create_internal_fence_pool();

    /**
	 * @brief Creates the command pool used by this device
	 */
    void create_internal_command_pool();

    /**
	 * @brief Requests a fence to the fence pool
	 * @return A vulkan fence
	 */
    VkFence request_fence() const;

    VkResult wait_idle() const;

    ResourceCache &get_resource_cache();

private:
    const PhysicalDevice &gpu;

    VkSurfaceKHR surface{VK_NULL_HANDLE};

    std::unique_ptr<DebugUtils> debug_utils;

    std::vector<VkExtensionProperties> device_extensions;

    std::vector<const char *> enabled_extensions{};

    VmaAllocator memory_allocator{VK_NULL_HANDLE};

    std::vector<std::vector<Queue>> queues;

    /// A command pool associated to the primary queue
    std::unique_ptr<CommandPool> command_pool;

    /// A fence pool associated to the primary queue
    std::unique_ptr<FencePool> fence_pool;

    ResourceCache resource_cache;
};

}// namespace vox::core
