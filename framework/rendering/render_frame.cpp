//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "render_frame.h"

#include "common/logging.h"

namespace vox::rendering {

RenderFrame::RenderFrame(core::Device &device,
                         std::unique_ptr<RenderTarget> &&render_target,
                         size_t thread_count)
    : core::FrameResource(device, thread_count),
      swapchain_render_target{std::move(render_target)} {
}

void RenderFrame::update_render_target(std::unique_ptr<RenderTarget> &&render_target) {
    swapchain_render_target = std::move(render_target);
}

RenderTarget &RenderFrame::get_render_target() {
    return *swapchain_render_target;
}

const RenderTarget &RenderFrame::get_render_target_const() const {
    return *swapchain_render_target;
}

}// namespace vox::rendering
