//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/image.h"
#include "core/vulkan_resource.h"

namespace vox {
namespace core {
class ImageView : public VulkanResource<VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW, const Device> {
public:
    ImageView(Image &image, VkImageViewType view_type, VkFormat format = VK_FORMAT_UNDEFINED,
              uint32_t base_mip_level = 0, uint32_t base_array_layer = 0,
              uint32_t n_mip_levels = 0, uint32_t n_array_layers = 0);

    ImageView(ImageView &) = delete;

    ImageView(ImageView &&other);

    ~ImageView() override;

    ImageView &operator=(const ImageView &) = delete;

    ImageView &operator=(ImageView &&) = delete;

    const Image &get_image() const;

    /**
	 * @brief Update the image this view is referring to
	 *        Used on image move
	 */
    void set_image(Image &image);

    VkFormat get_format() const;

    VkImageSubresourceRange get_subresource_range() const;

    VkImageSubresourceLayers get_subresource_layers() const;

private:
    Image *image{};

    VkFormat format{};

    VkImageSubresourceRange subresource_range{};
};

}
}// namespace vox::core
