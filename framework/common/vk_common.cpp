/* Copyright (c) 2018-2023, Arm Limited and Contributors
 * Copyright (c) 2019-2023, Sascha Willems
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vk_common.h"

#include <spdlog/fmt/fmt.h>

//#include "glsl_compiler.h"
//#include "platform/filesystem.h"

std::ostream &operator<<(std::ostream &os, const vk::Result result) {
#define WRITE_VK_ENUM(r)   \
    case vk::Result::e##r: \
        os << #r;          \
        break;

    switch (result) {
        WRITE_VK_ENUM(NotReady);
        WRITE_VK_ENUM(Timeout);
        WRITE_VK_ENUM(EventSet);
        WRITE_VK_ENUM(EventReset);
        WRITE_VK_ENUM(Incomplete);
        WRITE_VK_ENUM(ErrorOutOfHostMemory);
        WRITE_VK_ENUM(ErrorOutOfDeviceMemory);
        WRITE_VK_ENUM(ErrorInitializationFailed);
        WRITE_VK_ENUM(ErrorDeviceLost);
        WRITE_VK_ENUM(ErrorMemoryMapFailed);
        WRITE_VK_ENUM(ErrorLayerNotPresent);
        WRITE_VK_ENUM(ErrorExtensionNotPresent);
        WRITE_VK_ENUM(ErrorFeatureNotPresent);
        WRITE_VK_ENUM(ErrorIncompatibleDriver);
        WRITE_VK_ENUM(ErrorTooManyObjects);
        WRITE_VK_ENUM(ErrorFormatNotSupported);
        WRITE_VK_ENUM(ErrorSurfaceLostKHR);
        WRITE_VK_ENUM(ErrorNativeWindowInUseKHR);
        WRITE_VK_ENUM(SuboptimalKHR);
        WRITE_VK_ENUM(ErrorOutOfDateKHR);
        WRITE_VK_ENUM(ErrorIncompatibleDisplayKHR);
        WRITE_VK_ENUM(ErrorValidationFailedEXT);
        WRITE_VK_ENUM(ErrorInvalidShaderNV);
        default:
            os << "UNKNOWN_ERROR";
    }

#undef WRITE_VK_ENUM

    return os;
}

namespace vox {
namespace {
vk::ShaderStageFlagBits find_shader_stage(const std::string &ext) {
    if (ext == "vert") {
        return vk::ShaderStageFlagBits::eVertex;
    } else if (ext == "frag") {
        return vk::ShaderStageFlagBits::eFragment;
    } else if (ext == "comp") {
        return vk::ShaderStageFlagBits::eCompute;
    } else if (ext == "geom") {
        return vk::ShaderStageFlagBits::eGeometry;
    } else if (ext == "tesc") {
        return vk::ShaderStageFlagBits::eTessellationControl;
    } else if (ext == "tese") {
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    } else if (ext == "rgen") {
        return vk::ShaderStageFlagBits::eRaygenKHR;
    } else if (ext == "rahit") {
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    } else if (ext == "rchit") {
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    } else if (ext == "rmiss") {
        return vk::ShaderStageFlagBits::eMissKHR;
    } else if (ext == "rint") {
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    } else if (ext == "rcall") {
        return vk::ShaderStageFlagBits::eCallableKHR;
    } else if (ext == "mesh") {
        return vk::ShaderStageFlagBits::eMeshEXT;
    } else if (ext == "task") {
        return vk::ShaderStageFlagBits::eTaskEXT;
    }

    throw std::runtime_error("File extension `" + ext + "` does not have a vulkan shader stage.");
}
}// namespace
bool is_depth_only_format(vk::Format format) {
    return format == vk::Format::eD16Unorm ||
           format == vk::Format::eD32Sfloat;
}

bool is_depth_stencil_format(vk::Format format) {
    return format == vk::Format::eD16UnormS8Uint ||
           format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD32SfloatS8Uint ||
           is_depth_only_format(format);
}

vk::Format get_suitable_depth_format(vk::PhysicalDevice physical_device, bool depth_only,
                                     const std::vector<vk::Format> &depth_format_priority_list) {
    vk::Format depth_format{vk::Format::eUndefined};

    for (auto &format : depth_format_priority_list) {
        if (depth_only && !is_depth_only_format(format)) {
            continue;
        }

        vk::FormatProperties properties;
        physical_device.getFormatProperties(format, &properties);

        // Format must support depth stencil attachment for optimal tiling
        if (properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            depth_format = format;
            break;
        }
    }

    if (depth_format != vk::Format::eUndefined) {
        LOGI("Depth format selected: {}", to_string(depth_format));
        return depth_format;
    }

    throw std::runtime_error("No suitable depth format could be determined");
}

bool is_dynamic_buffer_descriptor_type(vk::DescriptorType descriptor_type) {
    return descriptor_type == vk::DescriptorType::eStorageBufferDynamic ||
           descriptor_type == vk::DescriptorType::eUniformBufferDynamic;
}

bool is_buffer_descriptor_type(vk::DescriptorType descriptor_type) {
    return descriptor_type == vk::DescriptorType::eStorageBuffer ||
           descriptor_type == vk::DescriptorType::eUniformBuffer ||
           is_dynamic_buffer_descriptor_type(descriptor_type);
}

int32_t get_bits_per_pixel(vk::Format format) {
    switch (format) {
        case vk::Format::eR4G4UnormPack8:
            return 8;
        case vk::Format::eR4G4B4A4UnormPack16:
        case vk::Format::eB4G4R4A4UnormPack16:
        case vk::Format::eR5G6B5UnormPack16:
        case vk::Format::eB5G6R5UnormPack16:
        case vk::Format::eR5G5B5A1UnormPack16:
        case vk::Format::eB5G5R5A1UnormPack16:
        case vk::Format::eA1R5G5B5UnormPack16:
            return 16;
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Snorm:
        case vk::Format::eR8Uscaled:
        case vk::Format::eR8Sscaled:
        case vk::Format::eR8Uint:
        case vk::Format::eR8Sint:
        case vk::Format::eR8Srgb:
            return 8;
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR8G8Uscaled:
        case vk::Format::eR8G8Sscaled:
        case vk::Format::eR8G8Uint:
        case vk::Format::eR8G8Sint:
        case vk::Format::eR8G8Srgb:
            return 16;
        case vk::Format::eR8G8B8Unorm:
        case vk::Format::eR8G8B8Snorm:
        case vk::Format::eR8G8B8Uscaled:
        case vk::Format::eR8G8B8Sscaled:
        case vk::Format::eR8G8B8Uint:
        case vk::Format::eR8G8B8Sint:
        case vk::Format::eR8G8B8Srgb:
        case vk::Format::eB8G8R8Unorm:
        case vk::Format::eB8G8R8Snorm:
        case vk::Format::eB8G8R8Uscaled:
        case vk::Format::eB8G8R8Sscaled:
        case vk::Format::eB8G8R8Uint:
        case vk::Format::eB8G8R8Sint:
        case vk::Format::eB8G8R8Srgb:
            return 24;
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Snorm:
        case vk::Format::eR8G8B8A8Uscaled:
        case vk::Format::eR8G8B8A8Sscaled:
        case vk::Format::eR8G8B8A8Uint:
        case vk::Format::eR8G8B8A8Sint:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Snorm:
        case vk::Format::eB8G8R8A8Uscaled:
        case vk::Format::eB8G8R8A8Sscaled:
        case vk::Format::eB8G8R8A8Uint:
        case vk::Format::eB8G8R8A8Sint:
        case vk::Format::eB8G8R8A8Srgb:
        case vk::Format::eA8B8G8R8UnormPack32:
        case vk::Format::eA8B8G8R8SnormPack32:
        case vk::Format::eA8B8G8R8UscaledPack32:
        case vk::Format::eA8B8G8R8SscaledPack32:
        case vk::Format::eA8B8G8R8UintPack32:
        case vk::Format::eA8B8G8R8SintPack32:
        case vk::Format::eA8B8G8R8SrgbPack32:
            return 32;
        case vk::Format::eA2R10G10B10UnormPack32:
        case vk::Format::eA2R10G10B10SnormPack32:
        case vk::Format::eA2R10G10B10UscaledPack32:
        case vk::Format::eA2R10G10B10SscaledPack32:
        case vk::Format::eA2R10G10B10UintPack32:
        case vk::Format::eA2R10G10B10SintPack32:
        case vk::Format::eA2B10G10R10UnormPack32:
        case vk::Format::eA2B10G10R10SnormPack32:
        case vk::Format::eA2B10G10R10UscaledPack32:
        case vk::Format::eA2B10G10R10SscaledPack32:
        case vk::Format::eA2B10G10R10UintPack32:
        case vk::Format::eA2B10G10R10SintPack32:
            return 32;
        case vk::Format::eR16Unorm:
        case vk::Format::eR16Snorm:
        case vk::Format::eR16Uscaled:
        case vk::Format::eR16Sscaled:
        case vk::Format::eR16Uint:
        case vk::Format::eR16Sint:
        case vk::Format::eR16Sfloat:
            return 16;
        case vk::Format::eR16G16Unorm:
        case vk::Format::eR16G16Snorm:
        case vk::Format::eR16G16Uscaled:
        case vk::Format::eR16G16Sscaled:
        case vk::Format::eR16G16Uint:
        case vk::Format::eR16G16Sint:
        case vk::Format::eR16G16Sfloat:
            return 32;
        case vk::Format::eR16G16B16Unorm:
        case vk::Format::eR16G16B16Snorm:
        case vk::Format::eR16G16B16Uscaled:
        case vk::Format::eR16G16B16Sscaled:
        case vk::Format::eR16G16B16Uint:
        case vk::Format::eR16G16B16Sint:
        case vk::Format::eR16G16B16Sfloat:
            return 48;
        case vk::Format::eR16G16B16A16Unorm:
        case vk::Format::eR16G16B16A16Snorm:
        case vk::Format::eR16G16B16A16Uscaled:
        case vk::Format::eR16G16B16A16Sscaled:
        case vk::Format::eR16G16B16A16Uint:
        case vk::Format::eR16G16B16A16Sint:
        case vk::Format::eR16G16B16A16Sfloat:
            return 64;
        case vk::Format::eR32Uint:
        case vk::Format::eR32Sint:
        case vk::Format::eR32Sfloat:
            return 32;
        case vk::Format::eR32G32Uint:
        case vk::Format::eR32G32Sint:
        case vk::Format::eR32G32Sfloat:
            return 64;
        case vk::Format::eR32G32B32Uint:
        case vk::Format::eR32G32B32Sint:
        case vk::Format::eR32G32B32Sfloat:
            return 96;
        case vk::Format::eR32G32B32A32Uint:
        case vk::Format::eR32G32B32A32Sint:
        case vk::Format::eR32G32B32A32Sfloat:
            return 128;
        case vk::Format::eR64Uint:
        case vk::Format::eR64Sint:
        case vk::Format::eR64Sfloat:
            return 64;
        case vk::Format::eR64G64Uint:
        case vk::Format::eR64G64Sint:
        case vk::Format::eR64G64Sfloat:
            return 128;
        case vk::Format::eR64G64B64Uint:
        case vk::Format::eR64G64B64Sint:
        case vk::Format::eR64G64B64Sfloat:
            return 192;
        case vk::Format::eR64G64B64A64Uint:
        case vk::Format::eR64G64B64A64Sint:
        case vk::Format::eR64G64B64A64Sfloat:
            return 256;
        case vk::Format::eB10G11R11UfloatPack32:
            return 32;
        case vk::Format::eE5B9G9R9UfloatPack32:
            return 32;
        case vk::Format::eD16Unorm:
            return 16;
        case vk::Format::eX8D24UnormPack32:
            return 32;
        case vk::Format::eD32Sfloat:
            return 32;
        case vk::Format::eS8Uint:
            return 8;
        case vk::Format::eD16UnormS8Uint:
            return 24;
        case vk::Format::eD24UnormS8Uint:
            return 32;
        case vk::Format::eD32SfloatS8Uint:
            return 40;
        case vk::Format::eUndefined:
        default:
            return -1;
    }
}

//vk::ShaderModule load_shader(const std::string &filename, vk::Device device, vk::ShaderStageFlagBits stage) {
//    vox::GLSLCompiler glsl_compiler;
//
//    auto buffer = vox::fs::read_shader_binary(filename);
//
//    std::string file_ext = filename;
//
//    // Extract extension name from the glsl shader file
//    file_ext = file_ext.substr(file_ext.find_last_of(".") + 1);
//
//    std::vector<uint32_t> spirv;
//    std::string info_log;
//
//    // Compile the GLSL source
//    if (!glsl_compiler.compile_to_spirv(vox::find_shader_stage(file_ext), buffer, "main", {}, spirv, info_log)) {
//        LOGE("Failed to compile shader, Error: {}", info_log.c_str());
//        return VK_NULL_HANDLE;
//    }
//
//    vk::ShaderModule shader_module;
//    vk::ShaderModuleCreateInfo module_create_info{};
//    module_create_info.sType = vk::StructureType::eShaderModuleCreateInfo;
//    module_create_info.codeSize = spirv.size() * sizeof(uint32_t);
//    module_create_info.pCode = spirv.data();
//
//    VK_CHECK(device.createShaderModule(&module_create_info, nullptr, &shader_module));
//
//    return shader_module;
//}

// Create an image memory barrier for changing the layout of
// an image and put it into an active command buffer
// See chapter 11.4 "Image Layout" for details

void set_image_layout(
    vk::CommandBuffer command_buffer,
    vk::Image image,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::ImageSubresourceRange subresource_range,
    vk::PipelineStageFlags src_mask,
    vk::PipelineStageFlags dst_mask) {
    // Create an image barrier object
    vk::ImageMemoryBarrier barrier{};
    barrier.sType = vk::StructureType::eImageMemoryBarrier;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.image = image;
    barrier.subresourceRange = subresource_range;

    // Source layouts (old)
    // Source access mask controls actions that have to be finished on the old layout
    // before it will be transitioned to the new layout
    switch (old_layout) {
        case vk::ImageLayout::eUndefined:
            // Image layout is undefined (or does not matter)
            // Only valid as initial layout
            // No flags required, listed only for completeness
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            break;

        case vk::ImageLayout::ePreinitialized:
            // Image is preinitialized
            // Only valid as initial layout for linear images, preserves memory contents
            // Make sure host writes have been finished
            barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
            break;

        case vk::ImageLayout::eColorAttachmentOptimal:
            // Image is a color attachment
            // Make sure any writes to the color buffer have been finished
            barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;

        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            // Image is a depth/stencil attachment
            // Make sure any writes to the depth/stencil buffer have been finished
            barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;

        case vk::ImageLayout::eTransferSrcOptimal:
            // Image is a transfer source
            // Make sure any reads from the image have been finished
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            break;

        case vk::ImageLayout::eTransferDstOptimal:
            // Image is a transfer destination
            // Make sure any writes to the image have been finished
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;

        case vk::ImageLayout::eReadOnlyOptimal:
            // Image is read by a shader
            // Make sure any shader reads from the image have been finished
            barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            // Other source layouts aren't handled (yet)
            break;
    }

    // Target layouts (new)
    // Destination access mask controls the dependency for the new image layout
    switch (new_layout) {
        case vk::ImageLayout::eTransferDstOptimal:
            // Image will be used as a transfer destination
            // Make sure any writes to the image have been finished
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;

        case vk::ImageLayout::eTransferSrcOptimal:
            // Image will be used as a transfer source
            // Make sure any reads from the image have been finished
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
            break;

        case vk::ImageLayout::eColorAttachmentOptimal:
            // Image will be used as a color attachment
            // Make sure any writes to the color buffer have been finished
            barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;

        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            // Image layout will be used as a depth/stencil attachment
            // Make sure any writes to depth/stencil buffer have been finished
            barrier.dstAccessMask = barrier.dstAccessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;

        case vk::ImageLayout::eShaderReadOnlyOptimal:
            // Image will be read in a shader (sampler, input attachment)
            // Make sure any writes to the image have been finished
            if (barrier.srcAccessMask == vk::AccessFlagBits::eNone) {
                barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite;
            }
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            // Other source layouts aren't handled (yet)
            break;
    }

    // Put barrier inside setup command buffer
    command_buffer.pipelineBarrier(
        src_mask,
        dst_mask,
        {},
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

// Fixed sub resource on first mip level and layer
void set_image_layout(
    vk::CommandBuffer command_buffer,
    vk::Image image,
    vk::ImageAspectFlags aspect_mask,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::PipelineStageFlags src_mask,
    vk::PipelineStageFlags dst_mask) {
    vk::ImageSubresourceRange subresource_range = {};
    subresource_range.aspectMask = aspect_mask;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.layerCount = 1;
    set_image_layout(command_buffer, image, old_layout, new_layout, subresource_range, src_mask, dst_mask);
}

void insert_image_memory_barrier(
    vk::CommandBuffer command_buffer,
    vk::Image image,
    vk::AccessFlags src_access_mask,
    vk::AccessFlags dst_access_mask,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::PipelineStageFlags src_stage_mask,
    vk::PipelineStageFlags dst_stage_mask,
    vk::ImageSubresourceRange subresource_range) {
    vk::ImageMemoryBarrier barrier{};
    barrier.sType = vk::StructureType::eImageMemoryBarrier;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcAccessMask = src_access_mask;
    barrier.dstAccessMask = dst_access_mask;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.image = image;
    barrier.subresourceRange = subresource_range;

    command_buffer.pipelineBarrier(
        src_stage_mask,
        dst_stage_mask,
        {},
        0, nullptr,
        0, nullptr,
        1, &barrier);
}
namespace gbuffer {
std::vector<LoadStoreInfo> get_load_all_store_swapchain() {
    // Load every attachment and store only swapchain
    std::vector<LoadStoreInfo> load_store{4};

    // Swapchain
    load_store[0].load_op = vk::AttachmentLoadOp::eDontCare;
    load_store[0].store_op = vk::AttachmentStoreOp::eStore;

    // Depth
    load_store[1].load_op = vk::AttachmentLoadOp::eLoad;
    load_store[1].store_op = vk::AttachmentStoreOp::eDontCare;

    // Albedo
    load_store[2].load_op = vk::AttachmentLoadOp::eLoad;
    load_store[2].store_op = vk::AttachmentStoreOp::eDontCare;

    // Normal
    load_store[3].load_op = vk::AttachmentLoadOp::eLoad;
    load_store[3].store_op = vk::AttachmentStoreOp::eDontCare;

    return load_store;
}

std::vector<LoadStoreInfo> get_clear_all_store_swapchain() {
    // Clear every attachment and store only swapchain
    std::vector<LoadStoreInfo> load_store{4};

    // Swapchain
    load_store[0].load_op = vk::AttachmentLoadOp::eClear;
    load_store[0].store_op = vk::AttachmentStoreOp::eDontCare;

    // Depth
    load_store[1].load_op = vk::AttachmentLoadOp::eClear;
    load_store[1].store_op = vk::AttachmentStoreOp::eDontCare;

    // Albedo
    load_store[2].load_op = vk::AttachmentLoadOp::eClear;
    load_store[2].store_op = vk::AttachmentStoreOp::eDontCare;

    // Normal
    load_store[3].load_op = vk::AttachmentLoadOp::eClear;
    load_store[3].store_op = vk::AttachmentStoreOp::eDontCare;

    return load_store;
}

std::vector<LoadStoreInfo> get_clear_store_all() {
    // Clear and store every attachment
    std::vector<LoadStoreInfo> load_store{4};

    // Swapchain
    load_store[0].load_op = vk::AttachmentLoadOp::eClear;
    load_store[0].store_op = vk::AttachmentStoreOp::eStore;

    // Depth
    load_store[1].load_op = vk::AttachmentLoadOp::eClear;
    load_store[1].store_op = vk::AttachmentStoreOp::eStore;

    // Albedo
    load_store[2].load_op = vk::AttachmentLoadOp::eClear;
    load_store[2].store_op = vk::AttachmentStoreOp::eStore;

    // Normal
    load_store[3].load_op = vk::AttachmentLoadOp::eClear;
    load_store[3].store_op = vk::AttachmentStoreOp::eStore;

    return load_store;
}

std::vector<vk::ClearValue> get_clear_value() {
    // Clear values
    std::vector<vk::ClearValue> clear_value{4};
    clear_value[0].color = {vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}};
    clear_value[1].depthStencil = {vk::ClearDepthStencilValue{0.0f, ~0U}};
    clear_value[2].color = {vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}};
    clear_value[3].color = {vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}};

    return clear_value;
}
}// namespace gbuffer

}// namespace vox
