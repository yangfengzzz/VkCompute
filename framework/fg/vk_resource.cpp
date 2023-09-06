//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "fg/vk_resource.h"
#include "fg/transient_resource_cache.h"
#include "core/barrier.h"

namespace vox::fg {
template<>
std::unique_ptr<core::Buffer> realize(TransientResourceCache &cache, const core::BufferDesc &description) {
    return cache.get_buffer(description);
}

template<>
void derealize(TransientResourceCache &cache, std::unique_ptr<core::Buffer> actual) {
    cache.insert_buffer(std::move(actual));
}

template<>
std::unique_ptr<core::Image> realize(TransientResourceCache &cache, const core::ImageDesc &description) {
    return cache.get_image(description);
}

template<>
void derealize(TransientResourceCache &cache, std::unique_ptr<core::Image> actual) {
    cache.insert_image(std::move(actual));
}

void set_barrier(core::CommandBuffer &cb, PassResource &pass) {
    auto image_resource = dynamic_cast<ImageResource *>(pass.handle);
    if (image_resource != nullptr) {
        core::record_image_barrier(cb, core::ImageBarrier{*image_resource->actual(),
                                                          image_resource->access_type(),
                                                          pass.access_type,
                                                          core::image_aspect_mask_from_access_type_and_format(pass.access_type,
                                                                                                              image_resource->actual()->get_format())});
    } else {
        auto buffer_resource = dynamic_cast<BufferResource *>(pass.handle);
        core::record_buffer_barrier(cb, core::BufferBarrier{*buffer_resource->actual(),
                                                            buffer_resource->access_type(),
                                                            pass.access_type});
    }
}

}// namespace vox::fg