//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/render_pass.h"
#include "rendering/render_target.h"

namespace vox {
namespace core {
class Device;
}// namespace core
namespace rendering {

class Framebuffer {
public:
    Framebuffer(core::Device &device, const RenderTarget &render_target, const core::RenderPass &render_pass);

    Framebuffer(const Framebuffer &) = delete;

    Framebuffer(Framebuffer &&other) noexcept;

    ~Framebuffer();

    Framebuffer &operator=(const Framebuffer &) = delete;

    Framebuffer &operator=(Framebuffer &&) = delete;

    [[nodiscard]] VkFramebuffer get_handle() const;

    [[nodiscard]] const VkExtent2D &get_extent() const;

private:
    core::Device &device;

    VkFramebuffer handle{VK_NULL_HANDLE};

    VkExtent2D extent{};
};

}// namespace rendering
}// namespace vox
