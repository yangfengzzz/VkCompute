//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/buffer.h"
#include "core/image.h"
#include "fg/render_task_base.h"
#include "fg/resource.h"

namespace vox::fg {
using BufferResource = Resource<core::BufferDesc, core::Buffer>;
using ImageResource = Resource<core::ImageDesc, core::Image>;

template<>
std::unique_ptr<core::Buffer> realize(const core::BufferDesc &description);

template<>
std::unique_ptr<core::Image> realize(const core::ImageDesc &description);

void set_barrier(core::CommandBuffer &cb, PassResource &pass);

}// namespace vox::fg