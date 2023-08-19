//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "core/frame_resource.h"

namespace vox::rendering {

/**
 * @brief RenderFrame is a container for per-frame data, including BufferPool objects,
 * synchronization primitives (semaphores, fences) and the swapchain RenderTarget.
 *
 * When creating a RenderTarget, we need to provide images that will be used as attachments
 * within a RenderPass. The RenderFrame is responsible for creating a RenderTarget using
 * RenderTarget::CreateFunc. A custom RenderTarget::CreateFunc can be provided if a different
 * render target is required.
 *
 * A RenderFrame cannot be destroyed individually since frames are managed by the RenderContext,
 * the whole context must be destroyed. This is because each RenderFrame holds Vulkan objects
 * such as the swapchain image.
 */
class RenderFrame : public core::FrameResource {
public:
    RenderFrame(core::Device &device, std::unique_ptr<RenderTarget> &&render_target, size_t thread_count = 1);

    RenderFrame(const RenderFrame &) = delete;

    RenderFrame(RenderFrame &&) = delete;

    RenderFrame &operator=(const RenderFrame &) = delete;

    RenderFrame &operator=(RenderFrame &&) = delete;

    /**
	 * @brief Called when the swapchain changes
	 * @param render_target A new render target with updated images
	 */
    void update_render_target(std::unique_ptr<RenderTarget> &&render_target);

    RenderTarget &get_render_target();

    [[nodiscard]] const RenderTarget &get_render_target_const() const;

private:
    std::unique_ptr<RenderTarget> swapchain_render_target;
};

}// namespace vox::rendering
