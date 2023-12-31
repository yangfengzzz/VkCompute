//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/utils.h"
#include "core/buffer.h"
#include "rendering/render_frame.h"
#include "rendering/subpass.h"

namespace vox::rendering {

/**
 * @brief A RenderPipeline is a sequence of Subpass objects.
 * Subpass holds shaders and can draw the sg::Scene.
 * More subpasses can be added to the sequence if required.
 * For example, postprocessing can be implemented with two pipelines which
 * share render targets.
 *
 * GeometrySubpass -> Processes Scene for Shaders, use by itself if shader requires no lighting
 * ForwardSubpass -> Binds lights at the beginning of a GeometrySubpass to create Forward Rendering, should be used with most default shaders
 * LightingSubpass -> Holds a Global Light uniform, Can be combined with GeometrySubpass to create Deferred Rendering
 */
class RenderPipeline {
public:
    explicit RenderPipeline(std::vector<std::unique_ptr<Subpass>> &&subpasses = {});

    RenderPipeline(const RenderPipeline &) = delete;

    RenderPipeline(RenderPipeline &&) = default;

    virtual ~RenderPipeline() = default;

    RenderPipeline &operator=(const RenderPipeline &) = delete;

    RenderPipeline &operator=(RenderPipeline &&) = default;

    /**
	 * @brief Prepares the subpasses
	 */
    void prepare();

    /**
	 * @return Load store info
	 */
    [[nodiscard]] const std::vector<LoadStoreInfo> &get_load_store() const;

    /**
	 * @param load_store Load store info to set
	 */
    void set_load_store(const std::vector<LoadStoreInfo> &load_store);

    /**
	 * @return Clear values
	 */
    [[nodiscard]] const std::vector<VkClearValue> &get_clear_value() const;

    /**
	 * @param clear_values Clear values to set
	 */
    void set_clear_value(const std::vector<VkClearValue> &clear_values);

    /**
	 * @brief Appends a subpass to the pipeline
	 * @param subpass Subpass to append
	 */
    void add_subpass(std::unique_ptr<Subpass> &&subpass);

    std::vector<std::unique_ptr<Subpass>> &get_subpasses();

    /**
	 * @brief Record draw commands for each Subpass
	 */
    void draw(core::CommandBuffer &command_buffer, RenderTarget &render_target, VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE);

    /**
	 * @return Subpass currently being recorded, or the first one
	 *         if drawing has not started
	 */
    std::unique_ptr<Subpass> &get_active_subpass();

private:
    std::vector<std::unique_ptr<Subpass>> subpasses;

    /// Default to two load store
    std::vector<LoadStoreInfo> load_store = std::vector<LoadStoreInfo>(2);

    /// Default to two clear values
    std::vector<VkClearValue> clear_value = std::vector<VkClearValue>(2);

    size_t active_subpass_index{0};
};

}// namespace vox::rendering
