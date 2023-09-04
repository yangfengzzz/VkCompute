//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "buffer.h"
#include "image.h"
#include <unordered_map>
#include <optional>

namespace vox::core {
class TransientResourceCache {
public:
    std::shared_ptr<Image> get_image(const ImageDesc &desc);

    void insert_image(std::shared_ptr<Image> image);

    std::shared_ptr<Buffer> get_buffer(const BufferDesc &desc);

    void insert_buffer(std::shared_ptr<Buffer> buffer);

private:
    std::unordered_map<ImageDesc, std::vector<std::shared_ptr<Image>>> images;
    std::unordered_map<BufferDesc, std::vector<std::shared_ptr<Buffer>>> buffers;
};
}// namespace vox::core