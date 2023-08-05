//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "sampled_image.h"

#include "rendering/render_target.h"

namespace vox {
namespace core {
SampledImage::SampledImage(const core::ImageView &image_view, Sampler *sampler) : image_view{&image_view},
                                                                                  target_attachment{0},
                                                                                  render_target{nullptr},
                                                                                  sampler{sampler},
                                                                                  isDepthResolve{false} {}

SampledImage::SampledImage(uint32_t target_attachment, RenderTarget *render_target, Sampler *sampler, bool isDepthResolve) : image_view{nullptr},
                                                                                                                             target_attachment{target_attachment},
                                                                                                                             render_target{render_target},
                                                                                                                             sampler{sampler},
                                                                                                                             isDepthResolve{isDepthResolve} {}

SampledImage::SampledImage(const SampledImage &to_copy) : image_view{to_copy.image_view},
                                                          target_attachment{to_copy.target_attachment},
                                                          render_target{to_copy.render_target},
                                                          sampler{to_copy.sampler},
                                                          isDepthResolve{false} {}

SampledImage &SampledImage::operator=(const SampledImage &to_copy) {
    image_view = to_copy.image_view;
    target_attachment = to_copy.target_attachment;
    render_target = to_copy.render_target;
    sampler = to_copy.sampler;
    isDepthResolve = to_copy.isDepthResolve;
    return *this;
}

SampledImage::SampledImage(SampledImage &&to_move) : image_view{std::move(to_move.image_view)},
                                                     target_attachment{std::move(to_move.target_attachment)},
                                                     render_target{std::move(to_move.render_target)},
                                                     sampler{std::move(to_move.sampler)},
                                                     isDepthResolve{std::move(to_move.isDepthResolve)} {}

SampledImage &SampledImage::operator=(SampledImage &&to_move) {
    image_view = std::move(to_move.image_view);
    target_attachment = std::move(to_move.target_attachment);
    render_target = std::move(to_move.render_target);
    sampler = std::move(to_move.sampler);
    isDepthResolve = std::move(to_move.isDepthResolve);
    return *this;
}

const core::ImageView &SampledImage::get_image_view(const RenderTarget &default_target) const {
    if (image_view != nullptr) {
        return *image_view;
    } else {
        const auto &target = render_target ? *render_target : default_target;
        assert(target_attachment < target.get_views().size());
        return target.get_views()[target_attachment];
    }
}

const uint32_t *SampledImage::get_target_attachment() const {
    if (image_view != nullptr) {
        return nullptr;
    } else {
        return &target_attachment;
    }
}

}
}// namespace vox::core