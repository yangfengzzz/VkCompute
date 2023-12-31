//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "fg/transient_resource_cache.h"

namespace vox::fg {
std::unique_ptr<core::Image> TransientResourceCache::get_image(const core::ImageDesc &desc) {
    auto iter = images.find(desc);
    if (iter != images.end()) {
        auto last = std::move(iter->second.back());
        iter->second.pop_back();
        return last;
    } else {
        return nullptr;
    }
}

void TransientResourceCache::insert_image(std::unique_ptr<core::Image> image) {
    auto desc = image->get_desc();
    auto iter = images.find(desc);
    if (iter != images.end()) {
        iter->second.push_back(std::move(image));
    } else {
        std::vector<std::unique_ptr<core::Image>> image_array{};
        image_array.push_back(std::move(image));
        images[desc] = std::move(image_array);
    }
}

std::unique_ptr<core::Buffer> TransientResourceCache::get_buffer(const core::BufferDesc &desc) {
    auto iter = buffers.find(desc);
    if (iter != buffers.end()) {
        auto last = std::move(iter->second.back());
        iter->second.pop_back();
        return last;
    } else {
        return nullptr;
    }
}

void TransientResourceCache::insert_buffer(std::unique_ptr<core::Buffer> buffer) {
    auto desc = buffer->get_desc();
    auto iter = buffers.find(desc);
    if (iter != buffers.end()) {
        iter->second.push_back(std::move(buffer));
    } else {
        std::vector<std::unique_ptr<core::Buffer>> buffer_array{};
        buffer_array.push_back(std::move(buffer));
        buffers[desc] = std::move(buffer_array);
    }
}
}// namespace vox::fg