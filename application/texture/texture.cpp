//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "texture/texture.h"

#include <mutex>

#include "common/error.h"

VKBP_DISABLE_WARNINGS()
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
VKBP_ENABLE_WARNINGS()

#include "framework/common/utils.h"
#include "texture/astc.h"
#include "texture/ktx.h"
#include "texture/stb.h"

namespace vox {
bool is_astc(const VkFormat format) {
    return (format == VK_FORMAT_ASTC_4x4_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_5x4_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_5x4_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_5x5_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_5x5_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_6x5_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_6x5_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_6x6_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_6x6_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_8x5_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_8x5_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_8x6_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_8x6_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_8x8_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_8x8_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_10x5_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_10x5_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_10x6_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_10x6_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_10x8_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_10x8_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_10x10_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_10x10_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_12x10_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_12x10_SRGB_BLOCK ||
            format == VK_FORMAT_ASTC_12x12_UNORM_BLOCK ||
            format == VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
}

// When the color-space of a loaded image is unknown (from KTX1 for example) we
// may want to assume that the loaded data is in sRGB format (since it usually is).
// In those cases, this helper will get called which will force an existing unorm
// format to become an srgb format where one exists. If none exist, the format will
// remain unmodified.
static VkFormat maybe_coerce_to_srgb(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_R8_UNORM:
            return VK_FORMAT_R8_SRGB;
        case VK_FORMAT_R8G8_UNORM:
            return VK_FORMAT_R8G8_SRGB;
        case VK_FORMAT_R8G8B8_UNORM:
            return VK_FORMAT_R8G8B8_SRGB;
        case VK_FORMAT_B8G8R8_UNORM:
            return VK_FORMAT_B8G8R8_SRGB;
        case VK_FORMAT_R8G8B8A8_UNORM:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
            return VK_FORMAT_A8B8G8R8_SRGB_PACK32;
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
            return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
            return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case VK_FORMAT_BC2_UNORM_BLOCK:
            return VK_FORMAT_BC2_SRGB_BLOCK;
        case VK_FORMAT_BC3_UNORM_BLOCK:
            return VK_FORMAT_BC3_SRGB_BLOCK;
        case VK_FORMAT_BC7_UNORM_BLOCK:
            return VK_FORMAT_BC7_SRGB_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
            return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
            return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
            return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
            return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
            return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
            return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
            return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
            return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
            return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
            return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
            return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
            return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
            return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
            return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
            return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
            return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
            return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG:
            return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG:
            return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG:
            return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG:
            return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;
        default:
            return fmt;
    }
}

Texture::Texture(const std::string &name, std::vector<uint8_t> &&d, std::vector<Mipmap> &&m)
    : name{name},
      data{std::move(d)},
      format{VK_FORMAT_R8G8B8A8_UNORM},
      mipmaps{std::move(m)} {
}

const std::vector<uint8_t> &Texture::get_data() const {
    return data;
}

void Texture::clear_data() {
    data.clear();
    data.shrink_to_fit();
}

VkFormat Texture::get_format() const {
    return format;
}

const VkExtent3D &Texture::get_extent() const {
    assert(!mipmaps.empty());
    return mipmaps[0].extent;
}

const uint32_t Texture::get_layers() const {
    return layers;
}

const std::vector<Mipmap> &Texture::get_mipmaps() const {
    return mipmaps;
}

const std::vector<std::vector<VkDeviceSize>> &Texture::get_offsets() const {
    return offsets;
}

void Texture::create_vk_image(core::Device const &device, VkImageViewType image_view_type, VkImageCreateFlags flags) {
    assert(!vk_image && !vk_image_view && "Vulkan image already constructed");

    vk_image = std::make_unique<core::Image>(device,
                                             get_extent(),
                                             format,
                                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                             VMA_MEMORY_USAGE_GPU_ONLY,
                                             VK_SAMPLE_COUNT_1_BIT,
                                             to_u32(mipmaps.size()),
                                             layers,
                                             VK_IMAGE_TILING_OPTIMAL,
                                             flags);
    vk_image->set_debug_name(name);

    vk_image_view = std::make_unique<core::ImageView>(*vk_image, image_view_type);
    vk_image_view->set_debug_name("View on " + name);
}

const core::Image &Texture::get_vk_image() const {
    assert(vk_image && "Vulkan image was not created");
    return *vk_image;
}

const core::ImageView &Texture::get_vk_image_view() const {
    assert(vk_image_view && "Vulkan image view was not created");
    return *vk_image_view;
}

Mipmap &Texture::get_mipmap(const size_t index) {
    assert(index < mipmaps.size());
    return mipmaps[index];
}

void Texture::generate_mipmaps() {
    assert(mipmaps.size() == 1 && "Mipmaps already generated");

    if (mipmaps.size() > 1) {
        return;// Do not generate again
    }

    auto extent = get_extent();
    auto next_width = std::max<uint32_t>(1u, extent.width / 2);
    auto next_height = std::max<uint32_t>(1u, extent.height / 2);
    auto channels = 4;
    auto next_size = next_width * next_height * channels;

    while (true) {
        // Make space for next mipmap
        auto old_size = to_u32(data.size());
        data.resize(old_size + next_size);

        auto &prev_mipmap = mipmaps.back();
        // Update mipmaps
        Mipmap next_mipmap{};
        next_mipmap.level = prev_mipmap.level + 1;
        next_mipmap.offset = old_size;
        next_mipmap.extent = {next_width, next_height, 1u};

        // Fill next mipmap memory
        stbir_resize_uint8(data.data() + prev_mipmap.offset, prev_mipmap.extent.width, prev_mipmap.extent.height, 0,
                           data.data() + next_mipmap.offset, next_mipmap.extent.width, next_mipmap.extent.height, 0, channels);

        mipmaps.emplace_back(std::move(next_mipmap));

        // Next mipmap values
        next_width = std::max<uint32_t>(1u, next_width / 2);
        next_height = std::max<uint32_t>(1u, next_height / 2);
        next_size = next_width * next_height * channels;

        if (next_width == 1 && next_height == 1) {
            break;
        }
    }
}

std::vector<Mipmap> &Texture::get_mut_mipmaps() {
    return mipmaps;
}

std::vector<uint8_t> &Texture::get_mut_data() {
    return data;
}

void Texture::set_data(const uint8_t *raw_data, size_t size) {
    assert(data.empty() && "Image data already set");
    data = {raw_data, raw_data + size};
}

void Texture::set_format(const VkFormat f) {
    format = f;
}

void Texture::set_width(const uint32_t width) {
    assert(!mipmaps.empty());
    mipmaps[0].extent.width = width;
}

void Texture::set_height(const uint32_t height) {
    assert(!mipmaps.empty());
    mipmaps[0].extent.height = height;
}

void Texture::set_depth(const uint32_t depth) {
    assert(!mipmaps.empty());
    mipmaps[0].extent.depth = depth;
}

void Texture::set_layers(uint32_t l) {
    layers = l;
}

void Texture::set_offsets(const std::vector<std::vector<VkDeviceSize>> &o) {
    offsets = o;
}

void Texture::coerce_format_to_srgb() {
    format = maybe_coerce_to_srgb(format);
}

std::unique_ptr<Texture> Texture::load(const std::string &name, const std::string &uri,
                                       ContentType content_type) {
    std::unique_ptr<Texture> image{nullptr};

    auto data = fs::read_asset(uri);

    // Get extension
    auto extension = get_extension(uri);

    if (extension == "png" || extension == "jpg") {
        image = std::make_unique<Stb>(name, data, content_type);
    } else if (extension == "astc") {
        image = std::make_unique<Astc>(name, data);
    } else if (extension == "ktx") {
        image = std::make_unique<Ktx>(name, data, content_type);
    } else if (extension == "ktx2") {
        image = std::make_unique<Ktx>(name, data, content_type);
    }

    return image;
}

}// namespace vox