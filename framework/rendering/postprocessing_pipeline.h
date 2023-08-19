//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "postprocessing_pass.h"

namespace vox::rendering {

class PostProcessingRenderPass;

/**
* @brief A rendering pipeline specialized for fullscreen post-processing and compute passes.
*/
class PostProcessingPipeline {
public:
    friend class PostProcessingPassBase;

    /**
    * @brief Creates a rendering pipeline entirely made of fullscreen post-processing subpasses.
    */
    PostProcessingPipeline(RenderContext &render_context, ShaderSource triangle_vs);

    PostProcessingPipeline(const PostProcessingPipeline &to_copy) = delete;
    PostProcessingPipeline &operator=(const PostProcessingPipeline &to_copy) = delete;

    PostProcessingPipeline(PostProcessingPipeline &&to_move) = delete;
    PostProcessingPipeline &operator=(PostProcessingPipeline &&to_move) = delete;

    virtual ~PostProcessingPipeline() = default;

    /**
	 * @brief Runs all renderpasses in this pipeline, recording commands into the given command buffer.
	 * @remarks vox::PostProcessingRenderpass that do not explicitly have a vox::RenderTarget set will render
	 *          to default_render_target.
	 */
    void draw(core::CommandBuffer &command_buffer, RenderTarget &default_render_target);

    /**
	 * @brief Gets all of the passes in the pipeline.
	 */
    inline std::vector<std::unique_ptr<PostProcessingPassBase>> &get_passes() {
        return passes;
    }

    /**
	 * @brief Get the pass at a certain index as a `TPass`.
	 */
    template<typename TPass = PostProcessingRenderPass>
    inline TPass &get_pass(size_t index) {
        return *dynamic_cast<TPass *>(passes[index].get());
    }

    /**
	 * @brief Adds a pass of the given type to the end of the pipeline by constructing it in-place.
	 */
    template<typename TPass = PostProcessingRenderPass, typename... ConstructorArgs>
    TPass &add_pass(ConstructorArgs &&...args) {
        passes.emplace_back(std::make_unique<TPass>(this, std::forward<ConstructorArgs>(args)...));
        auto &added_pass = *dynamic_cast<TPass *>(passes.back().get());
        return added_pass;
    }

    /**
	 * @brief Returns the current render context.
	 */
    [[nodiscard]] inline RenderContext &get_render_context() const {
        return *render_context;
    }

    /**
	 * @brief Returns the index of the currently-being-drawn pass.
	 */
    [[nodiscard]] inline size_t get_current_pass_index() const {
        return current_pass_index;
    }

private:
    RenderContext *render_context{nullptr};
    ShaderSource triangle_vs;
    std::vector<std::unique_ptr<PostProcessingPassBase>> passes{};
    size_t current_pass_index{0};
};

}// namespace vox::rendering