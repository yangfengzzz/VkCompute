//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "transient_resource_cache.h"

namespace vox::core {
std::shared_ptr<Image> TransientResourceCache::get_image(const ImageDesc &desc) {
    auto iter = images.find(desc);
    if (iter != images.end()) {
        auto last = iter->second.back();
        iter->second.pop_back();
        return last;
    } else {
        return nullptr;
    }
}

void TransientResourceCache::insert_image(std::shared_ptr<Image> image) {
    auto desc = image->get_desc();
    auto iter = images.find(desc);
    if (iter != images.end()) {
        iter->second.push_back(std::move(image));
    } else {
        std::vector<std::shared_ptr<Image>> image_array{};
        image_array.push_back(std::move(image));
        images.insert(std::make_pair(desc, image_array));
    }
}

std::shared_ptr<Buffer> TransientResourceCache::get_buffer(const BufferDesc &desc) {
    auto iter = buffers.find(desc);
    if (iter != buffers.end()) {
        auto last = iter->second.back();
        iter->second.pop_back();
        return last;
    } else {
        return nullptr;
    }
}

void TransientResourceCache::insert_buffer(std::shared_ptr<Buffer> buffer) {
    auto desc = buffer->get_desc();
    auto iter = buffers.find(desc);
    if (iter != buffers.end()) {
        iter->second.push_back(std::move(buffer));
    } else {
        std::vector<std::shared_ptr<Buffer>> buffer_array{};
        buffer_array.push_back(std::move(buffer));
        buffers.insert(std::make_pair(desc, buffer_array));
    }
}
}// namespace vox::core