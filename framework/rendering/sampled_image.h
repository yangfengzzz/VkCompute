//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/image.h"
#include "core/sampler.h"
#include <memory>

namespace vox {
namespace rendering {

class RenderTarget;
/**
* @brief A reference to a vox::ImageView, plus an optional sampler for it
*        - either coming from a vox::RenderTarget or from a user-created Image.
*/
class SampledImage {
public:
    /**
	* @brief Constructs a SampledImage referencing the given image and with the given sampler.
	* @remarks If the sampler is null, a default sampler will be used.
	*/
    explicit SampledImage(const core::ImageView &image_view, core::Sampler *sampler = nullptr);

    /**
	* @brief Constructs a SampledImage referencing a certain attachment of a render target.
	* @remarks If the render target is null, the default is assumed.
	*          If the sampler is null, a default sampler is used.
	*/
    explicit SampledImage(uint32_t target_attachment, RenderTarget *render_target = nullptr,
                          core::Sampler *sampler = nullptr, bool isDepthResolve = false);

    SampledImage(const SampledImage &to_copy);
    SampledImage &operator=(const SampledImage &to_copy);

    SampledImage(SampledImage &&to_move) noexcept;
    SampledImage &operator=(SampledImage &&to_move) noexcept;

    ~SampledImage() = default;

    /**
	 * @brief Replaces the current image view with the given one.
	 */
    inline void set_image_view(const core::ImageView &new_view) {
        image_view = &new_view;
    }

    /**
	 * @brief Replaces the image view with an attachment of the PostProcessingPipeline's render target.
	 */
    inline void set_image_view(uint32_t new_attachment) {
        image_view = nullptr;
        target_attachment = new_attachment;
    }

    /**
	 * @brief If this view refers to a render target attachment, returns a pointer to its index;
	 *        otherwise, returns `null`.
	 * @remarks The lifetime of the returned pointer matches that of this `SampledImage`.
	 */
    [[nodiscard]] const uint32_t *get_target_attachment() const;

    /**
	 * @brief Returns either the ImageView, if set, or the image view for the set target attachment.
	 *        If the view has no render target associated with it, default_target is used.
	 */
    [[nodiscard]] const core::ImageView &get_image_view(const RenderTarget &default_target) const;

    /**
	 * @brief Returns the currently-set sampler, if any.
	 */
    [[nodiscard]] inline core::Sampler *get_sampler() const {
        return sampler;
    }

    /**
	 * @brief Sets the sampler for this SampledImage.
	 */
    inline void set_sampler(core::Sampler *new_sampler) {
        sampler = new_sampler;
    }

    /**
	 * @brief Returns the RenderTarget, if set.
	 */
    [[nodiscard]] inline RenderTarget *get_render_target() const {
        return render_target;
    }

    /**
	 * @brief Returns either the RenderTarget, if set, or - if not - the given fallback render target.
	 */
    inline RenderTarget &get_render_target(RenderTarget &fallback) const {
        return render_target ? *render_target : fallback;
    }

    /**
	 * @brief Sets the sampler for this SampledImage.
	 *        Setting it to null will make it use the default instead.
	 */
    inline void set_render_target(RenderTarget *new_render_target) {
        render_target = new_render_target;
    }

    [[nodiscard]] inline bool is_depth_resolve() const {
        return isDepthResolve;
    }

private:
    const core::ImageView *image_view;
    uint32_t target_attachment;
    RenderTarget *render_target;
    core::Sampler *sampler;
    bool isDepthResolve;
};

}
}// namespace vox::rendering