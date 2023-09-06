//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/buffer.h"
#include "core/image.h"
#include <unordered_map>
#include <optional>

namespace vox::fg {
class TransientResourceCache {
public:
    std::unique_ptr<core::Image> get_image(const core::ImageDesc &desc);

    void insert_image(std::unique_ptr<core::Image> image);

    std::unique_ptr<core::Buffer> get_buffer(const core::BufferDesc &desc);

    void insert_buffer(std::unique_ptr<core::Buffer> buffer);

private:
    std::unordered_map<core::ImageDesc, std::vector<std::unique_ptr<core::Image>>> images;
    std::unordered_map<core::BufferDesc, std::vector<std::unique_ptr<core::Buffer>>> buffers;
};
}// namespace vox::fg