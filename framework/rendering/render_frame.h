//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "core/resource_caching.h"
#include "common/vk_common.h"
#include "core/buffer_pool.h"
#include "core/buffer.h"
#include "core/command_buffer.h"
#include "core/command_pool.h"
#include "core/device.h"
#include "core/image.h"
#include "core/query_pool.h"
#include "core/queue.h"
#include "core/fence_pool.h"
#include "core/semaphore_pool.h"
#include "rendering/render_target.h"

namespace vox {
namespace rendering {

enum BufferAllocationStrategy {
    OneAllocationPerBuffer,
    MultipleAllocationsPerBuffer
};

enum DescriptorManagementStrategy {
    StoreInCache,
    CreateDirectly
};

/**
 * @brief RenderFrame is a container for per-frame data, including BufferPool objects,
 * synchronization primitives (semaphores, fences) and the swapchain RenderTarget.
 *
 * When creating a RenderTarget, we need to provide images that will be used as attachments
 * within a RenderPass. The RenderFrame is responsible for creating a RenderTarget using
 * RenderTarget::CreateFunc. A custom RenderTarget::CreateFunc can be provided if a different
 * render target is required.
 *
 * A RenderFrame cannot be destroyed individually since frames are managed by the RenderContext,
 * the whole context must be destroyed. This is because each RenderFrame holds Vulkan objects
 * such as the swapchain image.
 */
class RenderFrame {
public:
    /**
	 * @brief Block size of a buffer pool in kilobytes
	 */
    static constexpr uint32_t BUFFER_POOL_BLOCK_SIZE = 256;

    // A map of the supported usages to a multiplier for the BUFFER_POOL_BLOCK_SIZE
    const std::unordered_map<VkBufferUsageFlags, uint32_t> supported_usage_map = {
        {VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 1},
        {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 2},// x2 the size of BUFFER_POOL_BLOCK_SIZE since SSBOs are normally much larger than other types of buffers
        {VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 1},
        {VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 1}};

    RenderFrame(core::Device &device, std::unique_ptr<RenderTarget> &&render_target, size_t thread_count = 1);

    RenderFrame(const RenderFrame &) = delete;

    RenderFrame(RenderFrame &&) = delete;

    RenderFrame &operator=(const RenderFrame &) = delete;

    RenderFrame &operator=(RenderFrame &&) = delete;

    void reset();

    core::Device &get_device();

    const core::FencePool &get_fence_pool() const;

    VkFence request_fence();

    const core::SemaphorePool &get_semaphore_pool() const;

    VkSemaphore request_semaphore();
    VkSemaphore request_semaphore_with_ownership();
    void release_owned_semaphore(VkSemaphore semaphore);

    /**
	 * @brief Called when the swapchain changes
	 * @param render_target A new render target with updated images
	 */
    void update_render_target(std::unique_ptr<RenderTarget> &&render_target);

    RenderTarget &get_render_target();

    const RenderTarget &get_render_target_const() const;

    /**
	 * @brief Requests a command buffer to the command pool of the active frame
	 *        A frame should be active at the moment of requesting it
	 * @param queue The queue command buffers will be submitted on
	 * @param reset_mode Indicate how the command buffer will be used, may trigger a
	 *        pool re-creation to set necessary flags
	 * @param level Command buffer level, either primary or secondary
	 * @param thread_index Selects the thread's command pool used to manage the buffer
	 * @return A command buffer related to the current active frame
	 */
    core::CommandBuffer &request_command_buffer(const core::Queue &queue,
                                                core::CommandBuffer::ResetMode reset_mode = core::CommandBuffer::ResetMode::ResetPool,
                                                VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                size_t thread_index = 0);

    VkDescriptorSet request_descriptor_set(const core::DescriptorSetLayout &descriptor_set_layout,
                                           const BindingMap<VkDescriptorBufferInfo> &buffer_infos,
                                           const BindingMap<VkDescriptorImageInfo> &image_infos,
                                           bool update_after_bind,
                                           size_t thread_index = 0);

    void clear_descriptors();

    /**
	 * @brief Sets a new buffer allocation strategy
	 * @param new_strategy The new buffer allocation strategy
	 */
    void set_buffer_allocation_strategy(BufferAllocationStrategy new_strategy);

    /**
	 * @brief Sets a new descriptor set management strategy
	 * @param new_strategy The new descriptor set management strategy
	 */
    void set_descriptor_management_strategy(DescriptorManagementStrategy new_strategy);

    /**
	 * @param usage Usage of the buffer
	 * @param size Amount of memory required
	 * @param thread_index Index of the buffer pool to be used by the current thread
	 * @return The requested allocation, it may be empty
	 */
    core::BufferAllocation allocate_buffer(VkBufferUsageFlags usage, VkDeviceSize size, size_t thread_index = 0);

    /**
	 * @brief Updates all the descriptor sets in the current frame at a specific thread index
	 */
    void update_descriptor_sets(size_t thread_index = 0);

private:
    core::Device &device;

    /**
	 * @brief Retrieve the frame's command pool(s)
	 * @param queue The queue command buffers will be submitted on
	 * @param reset_mode Indicate how the command buffers will be reset after execution,
	 *        may trigger a pool re-creation to set necessary flags
	 * @return The frame's command pool(s)
	 */
    std::vector<std::unique_ptr<core::CommandPool>> &get_command_pools(const core::Queue &queue, core::CommandBuffer::ResetMode reset_mode);

    /// Commands pools associated to the frame
    std::map<uint32_t, std::vector<std::unique_ptr<core::CommandPool>>> command_pools;

    /// Descriptor pools for the frame
    std::vector<std::unique_ptr<std::unordered_map<std::size_t, core::DescriptorPool>>> descriptor_pools;

    /// Descriptor sets for the frame
    std::vector<std::unique_ptr<std::unordered_map<std::size_t, core::DescriptorSet>>> descriptor_sets;

    core::FencePool fence_pool;

    core::SemaphorePool semaphore_pool;

    size_t thread_count;

    std::unique_ptr<RenderTarget> swapchain_render_target;

    BufferAllocationStrategy buffer_allocation_strategy{BufferAllocationStrategy::MultipleAllocationsPerBuffer};
    DescriptorManagementStrategy descriptor_management_strategy{DescriptorManagementStrategy::StoreInCache};

    std::map<VkBufferUsageFlags, std::vector<std::pair<core::BufferPool, core::BufferBlock *>>> buffer_pools;

    static std::vector<uint32_t> collect_bindings_to_update(const core::DescriptorSetLayout &descriptor_set_layout,
                                                            const BindingMap<VkDescriptorBufferInfo> &buffer_infos,
                                                            const BindingMap<VkDescriptorImageInfo> &image_infos);
};

}
}// namespace vox::rendering
