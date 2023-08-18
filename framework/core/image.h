//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unordered_set>

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/vulkan_resource.h"

namespace vox {
namespace core {
class Device;
class ImageView;

class Image : public VulkanResource<VkImage, VK_OBJECT_TYPE_IMAGE, const Device> {
public:
    Image(Device const &device,
          VkImage handle,
          const VkExtent3D &extent,
          VkFormat format,
          VkImageUsageFlags image_usage,
          VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT);

    Image(Device const &device,
          const VkExtent3D &extent,
          VkFormat format,
          VkImageUsageFlags image_usage,
          VmaMemoryUsage memory_usage,
          VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT,
          uint32_t mip_levels = 1,
          uint32_t array_layers = 1,
          VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL,
          VkImageCreateFlags flags = 0,
          uint32_t num_queue_families = 0,
          const uint32_t *queue_families = nullptr);

    Image(const Image &) = delete;

    Image(Image &&other);

    ~Image() override;

    Image &operator=(const Image &) = delete;

    Image &operator=(Image &&) = delete;

    VmaAllocation get_memory() const;

    /**
	 * @brief Maps vulkan memory to an host visible address
	 * @return Pointer to host visible memory
	 */
    uint8_t *map();

    /**
	 * @brief Unmaps vulkan memory from the host visible address
	 */
    void unmap();

    VkImageType get_type() const;

    const VkExtent3D &get_extent() const;

    VkFormat get_format() const;

    VkSampleCountFlagBits get_sample_count() const;

    VkImageUsageFlags get_usage() const;

    VkImageTiling get_tiling() const;

    VkImageSubresource get_subresource() const;

    uint32_t get_array_layer_count() const;

    std::unordered_set<ImageView *> &get_views();

private:
    VmaAllocation memory{VK_NULL_HANDLE};

    VkImageType type{};

    VkExtent3D extent{};

    VkFormat format{};

    VkImageUsageFlags usage{};

    VkSampleCountFlagBits sample_count{};

    VkImageTiling tiling{};

    VkImageSubresource subresource{};

    uint32_t array_layer_count{0};

    /// Image views referring to this image
    std::unordered_set<ImageView *> views;

    uint8_t *mapped_data{nullptr};

    /// Whether it was mapped with vmaMapMemory
    bool mapped{false};
};

}
}// namespace vox::core
