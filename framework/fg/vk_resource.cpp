//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "fg/vk_resource.h"
#include "core/barrier.h"

namespace vox::fg {
template<>
std::unique_ptr<core::Buffer> realize(const core::BufferDesc &description) {
    return nullptr;
}

template<>
std::unique_ptr<core::Image> realize(const core::ImageDesc &description) {
    return nullptr;
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