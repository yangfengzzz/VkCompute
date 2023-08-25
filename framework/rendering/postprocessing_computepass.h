//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/sampled_image.h"
#include "shader/shader_module.h"
#include "shader/shader_data.h"
#include "rendering/postprocessing_pass.h"

namespace vox::rendering {
/**
* @brief A compute pass in a vox::PostProcessingPipeline.
*/
class PostProcessingComputePass : public PostProcessingPass<PostProcessingComputePass> {
public:
    PostProcessingComputePass(PostProcessingPipeline *parent,
                              std::shared_ptr<ShaderModule> cs_source);

    PostProcessingComputePass(const PostProcessingComputePass &to_copy) = delete;
    PostProcessingComputePass &operator=(const PostProcessingComputePass &to_copy) = delete;

    PostProcessingComputePass(PostProcessingComputePass &&to_move) = default;
    PostProcessingComputePass &operator=(PostProcessingComputePass &&to_move) = default;

    void prepare(core::CommandBuffer &command_buffer, RenderTarget &default_render_target) override;

    void draw(core::CommandBuffer &command_buffer, RenderTarget &default_render_target) override;

public:
    /**
	 * @brief Sets the number of workgroups to be dispatched each draw().
	 */
    inline PostProcessingComputePass &set_dispatch_size(std::array<uint32_t, 3> new_size) {
        n_workgroups = new_size;
        return *this;
    }

    /**
	 * @brief Gets the number of workgroups that will be dispatched each draw().
	 */
    [[nodiscard]] inline std::array<uint32_t, 3> get_dispatch_size() const {
        return n_workgroups;
    }

    void attach_shader_data(ShaderData *data);

    void detach_shader_data(ShaderData *data);

    /**
	 * @brief Set the constants that are pushed before each draw.
	 */
    template<typename T>
    inline PostProcessingComputePass &set_push_constants(const T &data) {
        push_constants_data.reserve(sizeof(data));
        auto data_ptr = reinterpret_cast<const uint8_t *>(&data);
        push_constants_data.assign(data_ptr, data_ptr + sizeof(data));

        return *this;
    }

    /**
	 * @copydoc set_push_constants(const T&)
	 */
    inline PostProcessingComputePass &set_push_constants(const std::vector<uint8_t> &data) {
        push_constants_data = data;

        return *this;
    }

private:
    std::shared_ptr<ShaderModule> cs_source;
    std::array<uint32_t, 3> n_workgroups{1, 1, 1};

    std::vector<ShaderData *> data_{};
    std::vector<uint8_t> push_constants_data{};

    /**
	 * @brief Transitions sampled_images (to SHADER_READ_ONLY_OPTIMAL)
	 *        and storage_images (to GENERAL) as appropriate.
	 */
    void transition_images(core::CommandBuffer &command_buffer, RenderTarget &default_render_target);

    [[nodiscard]] BarrierInfo get_src_barrier_info() const override;
    [[nodiscard]] BarrierInfo get_dst_barrier_info() const override;
};

}// namespace vox::rendering