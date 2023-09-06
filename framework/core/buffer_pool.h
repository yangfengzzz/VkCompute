//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/vk_common.h"

namespace vox::core {
class Device;

class BufferPool {
public:
    BufferPool(core::Device &device, VkDeviceSize block_size, VkBufferUsageFlags usage,
               VkExportMemoryAllocateInfoKHR *info = nullptr);

    VmaPool get_handle() {
        return pool;
    }

    [[nodiscard]] bool is_exported() const {
        return is_exported_;
    }

    ~BufferPool();

private:
    Device &device;
    VmaPool pool{};
    bool is_exported_{false};
};
}// namespace vox::core