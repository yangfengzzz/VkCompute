//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/command_buffer.h"

namespace vox::core {
class Device;
class FrameResource;

class CommandPool {
public:
    CommandPool(Device &device, uint32_t queue_family_index,
                FrameResource *frame_resource = nullptr,
                size_t thread_index = 0,
                CommandBuffer::ResetMode reset_mode = CommandBuffer::ResetMode::ResetPool);

    CommandPool(const CommandPool &) = delete;

    CommandPool(CommandPool &&other) noexcept;

    ~CommandPool();

    CommandPool &operator=(const CommandPool &) = delete;

    CommandPool &operator=(CommandPool &&) = delete;

    Device &get_device();

    [[nodiscard]] uint32_t get_queue_family_index() const;

    [[nodiscard]] VkCommandPool get_handle() const;

    FrameResource *get_frame_resource();

    [[nodiscard]] size_t get_thread_index() const;

    VkResult reset_pool();

    CommandBuffer &request_command_buffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    [[nodiscard]] CommandBuffer::ResetMode get_reset_mode() const;

private:
    Device &device;

    VkCommandPool handle{VK_NULL_HANDLE};

    FrameResource *frame_resource{nullptr};

    size_t thread_index{0};

    uint32_t queue_family_index{0};

    std::vector<std::unique_ptr<CommandBuffer>> primary_command_buffers;

    uint32_t active_primary_command_buffer_count{0};

    std::vector<std::unique_ptr<CommandBuffer>> secondary_command_buffers;

    uint32_t active_secondary_command_buffer_count{0};

    CommandBuffer::ResetMode reset_mode{CommandBuffer::ResetMode::ResetPool};

    VkResult reset_command_buffers();
};

}// namespace vox::core
