//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "compute/compute_resource.h"

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
	 */
    explicit ComputeContext();

    ComputeContext(const ComputeContext &) = delete;

    ComputeContext(ComputeContext &&) = delete;

    virtual ~ComputeContext() = default;

    ComputeContext &operator=(const ComputeContext &) = delete;

    ComputeContext &operator=(ComputeContext &&) = delete;

    void create_device(core::PhysicalDevice& gpu);

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
    std::unordered_map<const char *, bool> get_instance_extensions();

    /**
	 * @brief Get sample-specific device extensions.
	 *
	 * @return Map of device extensions and whether or not they are optional. Default is empty map.
	 */
    std::unordered_map<const char *, bool> get_device_extensions();

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
    uint32_t get_device_count();

    ComputeResource get_resource(uint32_t index, size_t thread_count);

private:
    std::unique_ptr<core::DebugUtils> debug_utils{nullptr};

    /**
	 * @brief The Vulkan instance
	 */
    std::unique_ptr<core::Instance> instance{nullptr};

    /**
	 * @brief The Vulkan device
	 */
    std::vector<std::unique_ptr<core::Device>> devices{};

    /** @brief Set of device extensions to be enabled for this example and whether they are optional (must be set in the derived constructor) */
    std::unordered_map<const char *, bool> device_extensions;

    /** @brief Set of instance extensions to be enabled for this example and whether they are optional (must be set in the derived constructor) */
    std::unordered_map<const char *, bool> instance_extensions;

    /** @brief The Vulkan API version to request for this sample at instance creation time */
    uint32_t api_version = VK_API_VERSION_1_0;
};

}// namespace vox::compute
