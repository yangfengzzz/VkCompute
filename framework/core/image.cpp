//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "image.h"

#include "device.h"
#include "image_view.h"

namespace vox::core {
namespace {
inline VkImageType find_image_type(VkExtent3D extent) {
    VkImageType result{};

    uint32_t dim_num{0};

    if (extent.width >= 1) {
        dim_num++;
    }

    if (extent.height >= 1) {
        dim_num++;
    }

    if (extent.depth > 1) {
        dim_num++;
    }

    switch (dim_num) {
        case 1:
            result = VK_IMAGE_TYPE_1D;
            break;
        case 2:
            result = VK_IMAGE_TYPE_2D;
            break;
        case 3:
            result = VK_IMAGE_TYPE_3D;
            break;
        default:
            throw std::runtime_error("No image type found.");
    }

    return result;
}
}// namespace

Image::Image(Device const &device,
             ImageDesc desc,
             uint32_t num_queue_families,
             const uint32_t *queue_families) : VulkanResource{VK_NULL_HANDLE, &device},
                                               type{find_image_type(desc.extent)},
                                               desc{desc} {
    assert(mip_levels > 0 && "Image should have at least one level");
    assert(array_layers > 0 && "Image should have at least one layer");

    subresource.mipLevel = desc.mip_levels;
    subresource.arrayLayer = desc.array_layers;

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.flags = desc.flags;
    image_info.imageType = type;
    image_info.format = desc.format;
    image_info.extent = desc.extent;
    image_info.mipLevels = desc.mip_levels;
    image_info.arrayLayers = desc.array_layers;
    image_info.samples = desc.sample_count;
    image_info.tiling = desc.tiling;
    image_info.usage = desc.image_usage;

    if (num_queue_families != 0) {
        image_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        image_info.queueFamilyIndexCount = num_queue_families;
        image_info.pQueueFamilyIndices = queue_families;
    }

    VmaAllocationCreateInfo memory_info{};
    memory_info.usage = desc.memory_usage;

    if (desc.image_usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) {
        memory_info.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    }

    auto result = vmaCreateImage(device.get_memory_allocator(),
                                 &image_info, &memory_info,
                                 &handle, &memory,
                                 nullptr);

    if (result != VK_SUCCESS) {
        throw VulkanException{result, "Cannot create Image"};
    }
}

Image::Image(Device const &device, VkImage handle, const VkExtent3D &extent, VkFormat format,
             VkImageUsageFlags image_usage, VkSampleCountFlagBits sample_count)
    : VulkanResource{handle, &device},
      type{find_image_type(extent)},
      desc{.extent = extent,
           .format = format,
           .image_usage = image_usage,
           .sample_count = sample_count} {
    subresource.mipLevel = 1;
    subresource.arrayLayer = 1;
}

Image::Image(Image &&other) noexcept
    : VulkanResource{std::move(other)},
      memory{other.memory},
      type{other.type},
      desc{other.desc},
      subresource{other.subresource},
      views(std::exchange(other.views, {})),
      mapped_data{other.mapped_data},
      mapped{other.mapped} {
    other.memory = VK_NULL_HANDLE;
    other.mapped_data = nullptr;
    other.mapped = false;

    // Update image views references to this image to avoid dangling pointers
    for (auto &view : views) {
        view->set_image(*this);
    }
}

Image::~Image() {
    if (handle != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
        unmap();
        vmaDestroyImage(device->get_memory_allocator(), handle, memory);
    }
}

VmaAllocation Image::get_memory() const {
    return memory;
}

uint8_t *Image::map() {
    if (!mapped_data) {
        if (desc.tiling != VK_IMAGE_TILING_LINEAR) {
            LOGW("Mapping image memory that is not linear")
        }
        VK_CHECK(vmaMapMemory(device->get_memory_allocator(), memory, reinterpret_cast<void **>(&mapped_data)));
        mapped = true;
    }
    return mapped_data;
}

void Image::unmap() {
    if (mapped) {
        vmaUnmapMemory(device->get_memory_allocator(), memory);
        mapped_data = nullptr;
        mapped = false;
    }
}

VkImageType Image::get_type() const {
    return type;
}

const VkExtent3D &Image::get_extent() const {
    return desc.extent;
}

VkFormat Image::get_format() const {
    return desc.format;
}

VkSampleCountFlagBits Image::get_sample_count() const {
    return desc.sample_count;
}

VkImageUsageFlags Image::get_usage() const {
    return desc.image_usage;
}

VkImageTiling Image::get_tiling() const {
    return desc.tiling;
}

VkImageSubresource Image::get_subresource() const {
    return subresource;
}

uint32_t Image::get_array_layer_count() const {
    return desc.array_layers;
}

std::unordered_set<ImageView *> &Image::get_views() {
    return views;
}

}// namespace vox::core
