//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "texture/texture_manager.h"
#include "framework/core/device.h"
#include "shader/shader_manager.h"

namespace vox {
TextureManager *TextureManager::get_singleton_ptr() { return ms_singleton; }

TextureManager &TextureManager::get_singleton() {
    assert(ms_singleton);
    return (*ms_singleton);
}

TextureManager::TextureManager(core::Device &device)
    : device_(device), shader_data_(device), sampler_create_info_{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO} {
    // Create a default sampler
    sampler_create_info_.magFilter = VK_FILTER_LINEAR;
    sampler_create_info_.minFilter = VK_FILTER_LINEAR;
    sampler_create_info_.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_create_info_.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info_.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info_.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info_.mipLodBias = 0.0f;
    sampler_create_info_.compareOp = VK_COMPARE_OP_NEVER;
    sampler_create_info_.minLod = 0.0f;
    // Max level-of-detail should match mip level count
    sampler_create_info_.maxLod = 0.0f;
    // Only enable anisotropic filtering if enabled on the device
    // Note that for simplicity, we will always be using max. available anisotropy level for the current device
    // This may have an impact on performance, esp. on lower-specced devices
    // In a real-world scenario the level of anisotropy should be a user setting or e.g. lowered for mobile devices by
    // default
    sampler_create_info_.maxAnisotropy = device.get_gpu().get_features().samplerAnisotropy ? (device.get_gpu().get_properties().limits.maxSamplerAnisotropy) : 1.0f;
    sampler_create_info_.anisotropyEnable = device.get_gpu().get_features().samplerAnisotropy;
    sampler_create_info_.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    sampler_ = std::make_unique<core::Sampler>(device, sampler_create_info_);
}

std::shared_ptr<Texture> TextureManager::load_texture(const std::string &file) {
    auto iter = image_pool_.find(file);
    if (iter != image_pool_.end()) {
        return iter->second;
    } else {
        auto image = vox::Texture::load(file, file);
        image->create_vk_image(device_);
        upload_texture(image.get());
        image_pool_.insert(std::make_pair(file, image));
        return image;
    }
}

std::shared_ptr<Texture> TextureManager::load_texture_array(const std::string &file) {
    auto iter = image_pool_.find(file);
    if (iter != image_pool_.end()) {
        return iter->second;
    } else {
        auto image = vox::Texture::load(file, file);
        image->create_vk_image(device_, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
        upload_texture(image.get());
        image_pool_.insert(std::make_pair(file, image));
        return image;
    }
}

std::shared_ptr<Texture> TextureManager::load_texture_cubemap(const std::string &file) {
    auto iter = image_pool_.find(file);
    if (iter != image_pool_.end()) {
        return iter->second;
    } else {
        auto image = vox::Texture::load(file, file);
        image->create_vk_image(device_, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
        upload_texture(image.get());
        image_pool_.insert(std::make_pair(file, image));
        return image;
    }
}

void TextureManager::upload_texture(vox::Texture *image) {
    const auto &queue = device_.get_queue_by_flags(VK_QUEUE_TRANSFER_BIT, 0);
    core::CommandBuffer &command_buffer = device_.request_command_buffer();
    command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    vox::core::Buffer stage_buffer{device_, core::BufferDesc{image->get_data().size(),
                                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                             VMA_MEMORY_USAGE_CPU_ONLY}};
    stage_buffer.update(image->get_data());

    // Setup buffer copy regions for each mip level
    std::vector<VkBufferImageCopy> buffer_copy_regions;

    const auto &mipmaps = image->get_mipmaps();
    const auto &layers = image->get_layers();
    const auto &offsets = image->get_offsets();

    for (uint32_t layer = 0; layer < layers; layer++) {
        for (size_t i = 0; i < mipmaps.size(); i++) {
            VkBufferImageCopy buffer_copy_region = {};
            buffer_copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            buffer_copy_region.imageSubresource.mipLevel = to_u32(i);
            buffer_copy_region.imageSubresource.baseArrayLayer = layer;
            buffer_copy_region.imageSubresource.layerCount = 1;
            buffer_copy_region.imageExtent.width = image->get_extent().width >> i;
            buffer_copy_region.imageExtent.height = image->get_extent().height >> i;
            buffer_copy_region.imageExtent.depth = 1;
            buffer_copy_region.bufferOffset = layers > 1 ? offsets[layer][i] : mipmaps[i].offset;

            buffer_copy_regions.push_back(buffer_copy_region);
        }
    }

    VkImageSubresourceRange subresource_range = {};
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = to_u32(mipmaps.size());
    subresource_range.layerCount = layers;

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    command_buffer.image_memory_barrier(image->get_vk_image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Copy mip levels from staging buffer
    command_buffer.copy_buffer_to_image(stage_buffer, image->get_vk_image(), buffer_copy_regions);

    // Change texture image layout to shader read after all mip levels have been copied
    command_buffer.image_memory_barrier(image->get_vk_image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    command_buffer.end();
    queue.submit(command_buffer, device_.request_fence());
    device_.get_fence_pool().wait();
    device_.get_fence_pool().reset();
}

void TextureManager::collect_garbage() {
    for (auto &image : image_pool_) {
        if (image.second.use_count() == 1) {
            image.second.reset();
        }
    }
}

// MARK: - PBR
std::shared_ptr<Texture> TextureManager::generate_ibl(const std::string &file, rendering::RenderContext &render_context) {
    auto iter = image_pool_.find(file + "ibl");
    if (iter != image_pool_.end()) {
        return iter->second;
    } else {
        auto &command_buffer = render_context.begin();
        command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        auto source = load_texture_cubemap(file);
        auto baker_mipmap_count = static_cast<uint32_t>(source->get_mipmaps().size());
        std::vector<Mipmap> mipmap = source->get_mipmaps();

        auto target = std::make_shared<Texture>(file + "ibl", std::vector<uint8_t>(), std::move(mipmap));
        target->set_layers(source->get_layers());
        target->set_format(source->get_format());
        target->create_vk_image(device_, 0, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

//        if (!pipeline_) {
//            pipeline_ = std::make_unique<rendering::PostProcessingPipeline>(render_context, nullptr);
//            ibl_pass_ = &pipeline_->add_pass<rendering::PostProcessingComputePass>(
//                ShaderManager::get_singleton().load_shader("base/ibl.comp", VK_SHADER_STAGE_COMPUTE_BIT));
//            ibl_pass_->attach_shader_data(&shader_data_);
//        }
        shader_data_.set_sampled_texture("environmentMap", source->get_vk_image_view(VK_IMAGE_VIEW_TYPE_CUBE),
                                         sampler_.get());
        uint32_t source_width = source->get_extent().width;
        shader_data_.set_data("textureSize", source_width);

        auto &render_frame = render_context.get_active_frame();
        for (uint32_t lod = 0; lod < baker_mipmap_count; lod++) {
            float lod_roughness = float(lod) / float(baker_mipmap_count - 1);// linear
            auto &buffer = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(float), VMA_MEMORY_USAGE_GPU_ONLY, 0);
            buffer.convert_and_update(lod_roughness);
            // shader_data_.set_data("lodRoughness", buffer);

            shader_data_.set_storage_texture("o_results", target->get_vk_image_view(VK_IMAGE_VIEW_TYPE_CUBE, lod, 0, 1, 0));

//            ibl_pass_->set_dispatch_size({(source_width + 8) / 8, (source_width + 8) / 8, 6});
//            pipeline_->draw(command_buffer, render_context.get_active_frame().get_render_target());
        }
        command_buffer.end();
        render_context.submit(command_buffer);

        image_pool_.insert(std::make_pair(file + "ibl", target));
        return target;
    }
}

SphericalHarmonics3 TextureManager::generate_sh(const std::string &file) {
    auto source = load_texture_cubemap(file);
    const auto &layers = source->get_layers();
    auto &offsets = source->get_offsets();
    uint32_t texture_size = source->get_extent().width;
    float texel_size = 2.f / static_cast<float>(texture_size);// convolution is in the space of [-1, 1]

    float solid_angle_sum = 0;
    const uint32_t kChannelLength = get_bits_per_pixel(source->get_format()) / 8;
    const uint32_t kChannelShift = kChannelLength / 4;
    const float kTotalColor = std::pow(256.f, static_cast<float>(kChannelShift)) - 1;
    SphericalHarmonics3 sh;
    for (uint32_t layer = 0; layer < layers; layer++) {
        uint64_t buffer_offset = offsets[layer][0];
        float v = texel_size * 0.5f - 1.f;
        for (uint32_t y = 0; y < texture_size; y++) {
            float u = texel_size * 0.5f - 1.f;
            for (uint32_t x = 0; x < texture_size; x++) {
                uint64_t data_offset = y * texture_size * kChannelLength + x * kChannelLength + buffer_offset;

                float r = 0.f;
                for (uint8_t i = 0; i < kChannelShift; i++) {
                    r += static_cast<float>(source->data[data_offset + i]) * std::pow(256.f, i);
                }

                float g = 0.f;
                for (uint8_t i = 0; i < kChannelShift; i++) {
                    g += static_cast<float>(source->data[data_offset + i + kChannelShift]) * std::pow(256.f, i);
                }

                float b = 0.f;
                for (uint8_t i = 0; i < kChannelShift; i++) {
                    b += static_cast<float>(source->data[data_offset + i + 2 * kChannelShift]) * std::pow(256.f, i);
                }
                Color color(r / kTotalColor, g / kTotalColor, b / kTotalColor, 0);

                Vector3F direction;
                switch (layer) {
                    case 0:// TextureCubeFace.PositiveX
                        direction.set(1, -v, -u);
                        break;
                    case 1:// TextureCubeFace.NegativeX
                        direction.set(-1, -v, u);
                        break;
                    case 2:// TextureCubeFace.PositiveY
                        direction.set(u, -1, -v);
                        break;
                    case 3:// TextureCubeFace.NegativeY
                        direction.set(u, 1, v);
                        break;
                    case 4:// TextureCubeFace.PositiveZ
                        direction.set(u, -v, 1);
                        break;
                    case 5:// TextureCubeFace.NegativeZ
                        direction.set(-u, -v, -1);
                        break;
                    default:
                        break;
                }

                /**
                 * dA = cos = S / r = 4 / r
                 * dw = dA / r2 = 4 / r / r2
                 */
                float solid_angle = 4.f / (direction.length() * direction.lengthSquared());
                solid_angle_sum += solid_angle;
                sh.addLight(direction.normalized(), color, solid_angle);
                u += texel_size;
            }
            v += texel_size;
        }
    }
    sh = sh * (static_cast<float>(4.0 * M_PI) / solid_angle_sum);
    return sh;
}

}// namespace vox
