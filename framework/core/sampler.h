//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/vulkan_resource.h"

namespace vox::core {
class Device;

/**
 * @brief Represents a Vulkan Sampler
 */
class Sampler : public VulkanResource<VkSampler, VK_OBJECT_TYPE_SAMPLER, const Device> {
public:
    /**
	 * @brief Creates a Vulkan Sampler
	 * @param d The device to use
	 * @param info Creation details
	 */
    Sampler(Device const &d, const VkSamplerCreateInfo &info);

    Sampler(const Sampler &) = delete;

    Sampler(Sampler &&sampler) noexcept;

    ~Sampler() override;

    Sampler &operator=(const Sampler &) = delete;

    Sampler &operator=(Sampler &&) = delete;
};

}// namespace vox::core
