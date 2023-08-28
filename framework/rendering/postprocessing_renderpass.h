//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/sampled_image.h"
#include "rendering/postprocessing_pass.h"
#include "rendering/render_pipeline.h"
#include "rendering/subpass.h"

namespace vox::rendering {

/**
 * @brief An utility struct for hashing pairs.
 */
struct PairHasher {
    template<typename TPair>
    size_t operator()(const TPair &pair) const {
        std::hash<decltype(pair.first)> hash1{};
        std::hash<decltype(pair.second)> hash2{};
        return hash1(pair.first) * 43 + pair.second;
    }
};

/**
 * @brief Maps in-shader binding names to indices into a RenderTarget's attachments.
 */
using AttachmentMap = std::unordered_map<std::string, uint32_t>;

/**
 * @brief Maps in-shader binding names to the SampledImage to bind.
 */
using SampledMap = std::unordered_map<std::string, core::SampledImage>;

/**
 * @brief Maps in-shader binding names to the ImageView to bind for storage images.
 */
using StorageImageMap = std::unordered_map<std::string, const core::ImageView *>;

/**
 * @brief A list of indices into a RenderTarget's attachments.
 */
using AttachmentList = std::vector<uint32_t>;

/**
 * @brief A set of indices into a RenderTarget's attachments.
 */
using AttachmentSet = std::unordered_set<uint32_t>;

class PostProcessingRenderPass;

/**
 * @brief A single step of a vox::PostProcessingRenderPass.
 */
class PostProcessingSubpass : public Subpass {
public:
    PostProcessingSubpass(PostProcessingRenderPass *parent, RenderContext &render_context,
                          std::shared_ptr<ShaderModule> triangle_vs,
                          std::shared_ptr<ShaderModule> fs);

    PostProcessingSubpass(const PostProcessingSubpass &to_copy) = delete;
    PostProcessingSubpass &operator=(const PostProcessingSubpass &to_copy) = delete;

    PostProcessingSubpass(PostProcessingSubpass &&to_move) noexcept;
    PostProcessingSubpass &operator=(PostProcessingSubpass &&to_move) = delete;

    ~PostProcessingSubpass() override = default;

    /**
	 * @brief Maps the names of input attachments in the shader to indices into the render target's images.
	 *        These are given as `subpassInput`s to the subpass, at set 0; they are bound automatically according to their name.
	 */
    [[nodiscard]] inline const AttachmentMap &get_input_attachments() const {
        return input_attachments;
    }

    /**
	 * @brief Maps the names of samplers in the shader to vox::SampledImage.
	 *        These are given as samplers to the subpass, at set 0; they are bound automatically according to their name.
	 * @remarks PostProcessingPipeline::get_sampler() is used as the default sampler if none is specified.
	 *          The RenderTarget for the current PostProcessingSubpass is used if none is specified for attachment images.
	 */
    [[nodiscard]] inline const SampledMap &get_sampled_images() const {
        return sampled_images;
    }

    /**
	 * @brief Maps the names of storage images in the shader to vox::ImageView.
	 *        These are given as image2D[Array] to the subpass, at set 0; they are bound automatically according to their name.
	 */
    [[nodiscard]] inline const StorageImageMap &get_storage_images() const {
        return storage_images;
    }

    /**
	 * @brief Changes the debug name of this Subpass.
	 */
    inline PostProcessingSubpass &set_debug_name(const std::string &name) {
        Subpass::set_debug_name(name);

        return *this;
    }

    /**
	 * @brief Changes (or adds) the input attachment at name for this step.
	 */
    PostProcessingSubpass &bind_input_attachment(const std::string &name, uint32_t new_input_attachment);

    /**
	 * @brief Changes (or adds) the sampled image at name for this step.
	 * @remarks If no RenderTarget is specifically set for the SampledImage,
	 *          it will default to sample in the RenderTarget currently bound for drawing in the parent PostProcessingRenderpass.
	 */
    PostProcessingSubpass &bind_sampled_image(const std::string &name, core::SampledImage &&new_image);

    /**
	 * @brief Changes (or adds) the storage image at name for this step.
	 */
    PostProcessingSubpass &bind_storage_image(const std::string &name, const core::ImageView &new_image);

    /**
	 * @brief Removes the sampled image at name for this step.
	 */
    void unbind_sampled_image(const std::string &name);

    /**
	 * @brief Set the constants that are pushed before each fullscreen draw.
	 */
    PostProcessingSubpass &set_push_constants(const std::vector<uint8_t> &data);

    /**
	 * @brief Set the constants that are pushed before each fullscreen draw.
	 */
    template<typename T>
    inline PostProcessingSubpass &set_push_constants(const T &data) {
        push_constants_data.reserve(sizeof(data));
        auto data_ptr = reinterpret_cast<const uint8_t *>(&data);
        push_constants_data.assign(data_ptr, data_ptr + sizeof(data));

        return *this;
    }

    /**
	 * @brief A functor used to draw the primitives for a post-processing step.
	 * @see default_draw_func()
	 */
    using DrawFunc = std::function<void(core::CommandBuffer &command_buffer, RenderTarget &render_target)>;

    /**
	 * @brief Sets the function used to draw this postprocessing step.
	 * @see default_draw_func()
	 */
    PostProcessingSubpass &set_draw_func(DrawFunc &&new_func);

    /**
	 * @brief The default function used to draw a step; it draws 1 instance with 3 vertices.
	 */
    static void default_draw_func(vox::core::CommandBuffer &command_buffer, RenderTarget &render_target);

private:
    PostProcessingRenderPass *parent;

    std::shared_ptr<ShaderModule> vertex_shader_{nullptr};
    std::shared_ptr<ShaderModule> fragment_shader_{nullptr};

    AttachmentMap input_attachments{};
    SampledMap sampled_images{};
    StorageImageMap storage_images{};

    std::vector<uint8_t> push_constants_data{};

    DrawFunc draw_func{&PostProcessingSubpass::default_draw_func};

    void prepare() override;
    void draw(core::CommandBuffer &command_buffer) override;
};

/**
 * @brief A collection of vox::PostProcessingSubpass that are run as a single renderpass.
 */
class PostProcessingRenderPass : public PostProcessingPass<PostProcessingRenderPass> {
public:
    friend class PostProcessingSubpass;

    explicit PostProcessingRenderPass(PostProcessingPipeline *parent, std::unique_ptr<core::Sampler> &&default_sampler = nullptr);

    PostProcessingRenderPass(const PostProcessingRenderPass &to_copy) = delete;
    PostProcessingRenderPass &operator=(const PostProcessingRenderPass &to_copy) = delete;

    PostProcessingRenderPass(PostProcessingRenderPass &&to_move) = default;
    PostProcessingRenderPass &operator=(PostProcessingRenderPass &&to_move) = default;

    void draw(core::CommandBuffer &command_buffer, RenderTarget &default_render_target) override;

    /**
	 * @brief Gets the step at the given index.
	 */
    inline PostProcessingSubpass &get_subpass(size_t index) {
        assert(index < pipeline.get_subpasses().size());
        return *dynamic_cast<PostProcessingSubpass *>(pipeline.get_subpasses()[index].get());
    }

    /**
	 * @brief Constructs a new PostProcessingSubpass and adds it to the tail of the pipeline.
	 * @remarks `this`, the render context and the vertex shader source are passed automatically before `args`.
	 * @returns The inserted step.
	 */
    template<typename... ConstructorArgs>
    PostProcessingSubpass &add_subpass(ConstructorArgs &&...args) {
        std::shared_ptr<ShaderModule> vs_copy = get_triangle_vs();
        auto new_subpass = std::make_unique<PostProcessingSubpass>(this, get_render_context(), std::move(vs_copy), std::forward<ConstructorArgs>(args)...);
        auto &new_subpass_ref = *new_subpass;

        pipeline.add_subpass(std::move(new_subpass));

        return new_subpass_ref;
    }

    /**
	 * @brief Set the uniform data to be bound at set 0, binding 0.
	 */
    template<typename T>
    inline PostProcessingRenderPass &set_uniform_data(const T &data) {
        uniform_data.reserve(sizeof(data));
        auto data_ptr = reinterpret_cast<const uint8_t *>(&data);
        uniform_data.assign(data_ptr, data_ptr + sizeof(data));

        return *this;
    }

    /**
	 * @copydoc set_uniform_data(const T&)
	 */
    inline PostProcessingRenderPass &set_uniform_data(const std::vector<uint8_t> &data) {
        uniform_data = data;

        return *this;
    }

private:
    // An attachment sampled from a rendertarget
    using SampledAttachmentSet = std::unordered_set<std::pair<RenderTarget *, uint32_t>, PairHasher>;

    /**
	 * @brief Transition input, sampled and output attachments as appropriate.
	 * @remarks If a RenderTarget is not explicitly set for this pass, fallback_render_target is used.
	 */
    void transition_attachments(const AttachmentSet &input_attachments,
                                const SampledAttachmentSet &sampled_attachments,
                                const AttachmentSet &output_attachments,
                                core::CommandBuffer &command_buffer,
                                RenderTarget &fallback_render_target);

    /**
	 * @brief Select appropriate load/store operations for each buffer of render_target,
	 *        according to the subpass inputs/sampled inputs/subpass outputs of all steps
	 *        in the pipeline.
	 * @remarks If a RenderTarget is not explicitly set for this pass, fallback_render_target is used.
	 */
    void update_load_stores(const AttachmentSet &input_attachments,
                            const SampledAttachmentSet &sampled_attachments,
                            const AttachmentSet &output_attachments,
                            const RenderTarget &fallback_render_target);

    /**
	 * @brief Transition images and prepare load/stores before draw()ing.
	 */
    void prepare_draw(core::CommandBuffer &command_buffer, RenderTarget &fallback_render_target);

    [[nodiscard]] BarrierInfo get_src_barrier_info() const override;
    [[nodiscard]] BarrierInfo get_dst_barrier_info() const override;

    RenderPipeline pipeline{};
    std::unique_ptr<core::Sampler> default_sampler{};
    RenderTarget *draw_render_target{nullptr};
    std::vector<LoadStoreInfo> load_stores{};
    bool load_stores_dirty{true};
    std::vector<uint8_t> uniform_data{};
};

}// namespace vox::rendering
