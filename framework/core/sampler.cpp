//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "sampler.h"

#include "device.h"

namespace vox::core {
Sampler::Sampler(Device const &d, const VkSamplerCreateInfo &info) : VulkanResource{VK_NULL_HANDLE, &d} {
    VK_CHECK(vkCreateSampler(device->get_handle(), &info, nullptr, &handle));
}

Sampler::Sampler(Sampler &&other) noexcept : VulkanResource{std::move(other)} {
}

Sampler::~Sampler() {
    if (handle != VK_NULL_HANDLE) {
        vkDestroySampler(device->get_handle(), handle, nullptr);
    }
}

}// namespace vox::core
