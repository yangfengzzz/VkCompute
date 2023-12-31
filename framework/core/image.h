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

namespace vox::core {
class Device;
class ImageView;

struct ImageDesc {
    VkExtent3D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags image_usage{};
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_UNKNOWN;
    VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageCreateFlags flags = 0;
};

inline bool operator==(const ImageDesc &x, const ImageDesc &y) {
    return x.format == y.format &&
           x.extent.width == y.extent.width &&
           x.extent.height == y.extent.height &&
           x.extent.depth == y.extent.depth &&
           x.image_usage == y.image_usage &&
           x.memory_usage == y.memory_usage &&
           x.sample_count == y.sample_count &&
           x.mip_levels == y.mip_levels &&
           x.array_layers == y.array_layers &&
           x.tiling == y.tiling &&
           x.flags == y.flags;
}

class Image : public VulkanResource<VkImage, VK_OBJECT_TYPE_IMAGE, const Device> {
public:
    Image(Device const &device,
          ImageDesc desc,
          uint32_t num_queue_families = 0,
          const uint32_t *queue_families = nullptr);

    Image(Device const &device,
          VkImage handle,
          const VkExtent3D &extent,
          VkFormat format,
          VkImageUsageFlags image_usage,
          VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT);

    Image(const Image &) = delete;

    Image(Image &&other) noexcept;

    ~Image() override;

    Image &operator=(const Image &) = delete;

    Image &operator=(Image &&) = delete;

    [[nodiscard]] VmaAllocation get_memory() const;

    /**
	 * @brief Maps vulkan memory to an host visible address
	 * @return Pointer to host visible memory
	 */
    [[nodiscard]] uint8_t *map();

    /**
	 * @brief Unmaps vulkan memory from the host visible address
	 */
    void unmap();

    [[nodiscard]] VkImageType get_type() const;

    [[nodiscard]] const VkExtent3D &get_extent() const;

    [[nodiscard]] VkFormat get_format() const;

    [[nodiscard]] VkSampleCountFlagBits get_sample_count() const;

    [[nodiscard]] VkImageUsageFlags get_usage() const;

    [[nodiscard]] VkImageTiling get_tiling() const;

    [[nodiscard]] VkImageSubresource get_subresource() const;

    [[nodiscard]] uint32_t get_array_layer_count() const;

    [[nodiscard]] std::unordered_set<ImageView *> &get_views();

    inline ImageDesc get_desc() { return desc; }

private:
    VmaAllocation memory{VK_NULL_HANDLE};

    VkImageType type{};

    ImageDesc desc{};

    VkImageSubresource subresource{};

    /// Image views referring to this image
    std::unordered_set<ImageView *> views;

    uint8_t *mapped_data{nullptr};

    /// Whether it was mapped with vmaMapMemory
    bool mapped{false};
};

}// namespace vox::core

namespace std {
template<>
struct hash<vox::core::ImageDesc> {
    std::size_t operator()(const vox::core::ImageDesc &desc) const {
        std::size_t result = 0;

        vox::hash_combine(result, desc.memory_usage);
        vox::hash_combine(result, desc.sample_count);
        vox::hash_combine(result, desc.image_usage);
        vox::hash_combine(result, desc.array_layers);
        vox::hash_combine(result, desc.format);
        vox::hash_combine(result, desc.extent.width);
        vox::hash_combine(result, desc.extent.height);
        vox::hash_combine(result, desc.extent.depth);
        vox::hash_combine(result, desc.flags);
        vox::hash_combine(result, desc.mip_levels);
        vox::hash_combine(result, desc.tiling);

        return result;
    }
};
}// namespace std