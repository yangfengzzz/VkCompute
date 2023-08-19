//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <volk.h>

#include "framework/core/image.h"
#include "framework/core/image_view.h"

namespace vox {
/**
 * @param format Vulkan format
 * @return Whether the vulkan format is ASTC
 */
bool is_astc(VkFormat format);

/**
 * @brief Mipmap information
 */
struct Mipmap {
    /// Mipmap level
    uint32_t level = 0;

    /// Byte offset used for uploading
    uint32_t offset = 0;

    /// Width depth and height of the mipmap
    VkExtent3D extent = {0, 0, 0};
};

class Texture {
public:
    std::string name;

    /**
		 * @brief Type of content held in image.
		 * This helps to steer the image loaders when deciding what the format should be.
		 * Some image containers don't know whether the data they contain is sRGB or not.
		 * Since most applications save color images in sRGB, knowing that an image
		 * contains color data helps us to better guess its format when unknown.
		 */
    enum ContentType {
        Unknown,
        Color,
        Other
    };

    explicit Texture(const std::string &name, std::vector<uint8_t> &&data = {}, std::vector<Mipmap> &&mipmaps = {{}});

    static std::shared_ptr<Texture> load(const std::string &name, const std::string &uri,
                                         ContentType content_type = Unknown);

    virtual ~Texture() = default;

    [[nodiscard]] const std::vector<uint8_t> &get_data() const;

    void clear_data();

    [[nodiscard]] VkFormat get_format() const;

    [[nodiscard]] const VkExtent3D &get_extent() const;

    [[nodiscard]] const uint32_t get_layers() const;

    [[nodiscard]] const std::vector<Mipmap> &get_mipmaps() const;

    [[nodiscard]] const std::vector<std::vector<VkDeviceSize>> &get_offsets() const;

    void generate_mipmaps();

    void create_vk_image(core::Device const &device,
                         VkImageCreateFlags flags = 0,
                         VkImageUsageFlags image_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    [[nodiscard]] const core::Image &get_vk_image() const;

    [[nodiscard]] const core::ImageView &get_vk_image_view(VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D,
                                                           uint32_t base_mip_level = 0,
                                                           uint32_t base_array_layer = 0,
                                                           uint32_t n_mip_levels = 0,
                                                           uint32_t n_array_layers = 0);

    void coerce_format_to_srgb();

protected:
    friend class TextureManager;

    std::vector<uint8_t> &get_mut_data();

    void set_data(const uint8_t *raw_data, size_t size);

    void set_format(VkFormat format);

    void set_width(uint32_t width);

    void set_height(uint32_t height);

    void set_depth(uint32_t depth);

    void set_layers(uint32_t layers);

    void set_offsets(const std::vector<std::vector<VkDeviceSize>> &offsets);

    Mipmap &get_mipmap(size_t index);

    std::vector<Mipmap> &get_mut_mipmaps();

private:
    std::vector<uint8_t> data;

    VkFormat format{VK_FORMAT_UNDEFINED};

    uint32_t layers{1};

    std::vector<Mipmap> mipmaps{{}};

    // Offsets stored like offsets[array_layer][mipmap_layer]
    std::vector<std::vector<VkDeviceSize>> offsets;

    std::unique_ptr<core::Image> vk_image;

    std::unordered_map<size_t, std::unique_ptr<core::ImageView>> vk_image_views;
};

}// namespace vox
