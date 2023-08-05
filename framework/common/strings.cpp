/* Copyright (c) 2018-2023, Arm Limited and Contributors
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

#include "strings.h"

#include <spdlog/fmt/fmt.h>

//#include "core/shader_module.h"
//#include "scene_graph/components/material.h"

namespace vox {
std::vector<std::string> split(const std::string &str, const std::string &delimiter) {
    if (str.size() == 0) {
        return {};
    }

    std::vector<std::string> out;

    std::string buffer = str;
    size_t last_found_pos = 0;
    size_t pos = 0;
    while ((pos = buffer.find(delimiter)) != std::string::npos) {
        out.push_back(buffer.substr(0, pos));
        buffer.erase(0, pos + delimiter.length());
        last_found_pos = last_found_pos + pos + delimiter.length();
    }

    if (last_found_pos == str.size()) {
        out.push_back("");
    }

    return out;
}

std::string join(const std::vector<std::string> &str, const std::string &separator) {
    std::stringstream out;

    for (auto it = str.begin(); it != str.end(); it++) {
        out << *it;

        if (it != str.end() - 1) {
            out << separator;
        }
    }

    return out.str();
}

const std::string to_string(vk::Format format) {
    switch (format) {
        case vk::Format::eR4G4UnormPack8:
            return "VK_FORMAT_R4G4_UNORM_PACK8";
        case vk::Format::eR4G4B4A4UnormPack16:
            return "VK_FORMAT_R4G4B4A4_UNORM_PACK16";
        case vk::Format::eB4G4R4A4UnormPack16:
            return "VK_FORMAT_B4G4R4A4_UNORM_PACK16";
        case vk::Format::eR5G6B5UnormPack16:
            return "VK_FORMAT_R5G6B5_UNORM_PACK16";
        case vk::Format::eB5G6R5UnormPack16:
            return "VK_FORMAT_B5G6R5_UNORM_PACK16";
        case vk::Format::eR5G5B5A1UnormPack16:
            return "VK_FORMAT_R5G5B5A1_UNORM_PACK16";
        case vk::Format::eB5G5R5A1UnormPack16:
            return "VK_FORMAT_B5G5R5A1_UNORM_PACK16";
        case vk::Format::eA1R5G5B5UnormPack16:
            return "VK_FORMAT_A1R5G5B5_UNORM_PACK16";
        case vk::Format::eR8Unorm:
            return "VK_FORMAT_R8_UNORM";
        case vk::Format::eR8Snorm:
            return "VK_FORMAT_R8_SNORM";
        case vk::Format::eR8Uscaled:
            return "VK_FORMAT_R8_USCALED";
        case vk::Format::eR8Sscaled:
            return "VK_FORMAT_R8_SSCALED";
        case vk::Format::eR8Uint:
            return "VK_FORMAT_R8_UINT";
        case vk::Format::eR8Sint:
            return "VK_FORMAT_R8_SINT";
        case vk::Format::eR8Srgb:
            return "VK_FORMAT_R8_SRGB";
        case vk::Format::eR8G8Unorm:
            return "VK_FORMAT_R8G8_UNORM";
        case vk::Format::eR8G8Snorm:
            return "VK_FORMAT_R8G8_SNORM";
        case vk::Format::eR8G8Uscaled:
            return "VK_FORMAT_R8G8_USCALED";
        case vk::Format::eR8G8Sscaled:
            return "VK_FORMAT_R8G8_SSCALED";
        case vk::Format::eR8G8Uint:
            return "VK_FORMAT_R8G8_UINT";
        case vk::Format::eR8G8Sint:
            return "VK_FORMAT_R8G8_SINT";
        case vk::Format::eR8G8Srgb:
            return "VK_FORMAT_R8G8_SRGB";
        case vk::Format::eR8G8B8Unorm:
            return "VK_FORMAT_R8G8B8_UNORM";
        case vk::Format::eR8G8B8Snorm:
            return "VK_FORMAT_R8G8B8_SNORM";
        case vk::Format::eR8G8B8Uscaled:
            return "VK_FORMAT_R8G8B8_USCALED";
        case vk::Format::eR8G8B8Sscaled:
            return "VK_FORMAT_R8G8B8_SSCALED";
        case vk::Format::eR8G8B8Uint:
            return "VK_FORMAT_R8G8B8_UINT";
        case vk::Format::eR8G8B8Sint:
            return "VK_FORMAT_R8G8B8_SINT";
        case vk::Format::eR8G8B8Srgb:
            return "VK_FORMAT_R8G8B8_SRGB";
        case vk::Format::eB8G8R8Unorm:
            return "VK_FORMAT_B8G8R8_UNORM";
        case vk::Format::eB8G8R8Snorm:
            return "VK_FORMAT_B8G8R8_SNORM";
        case vk::Format::eB8G8R8Uscaled:
            return "VK_FORMAT_B8G8R8_USCALED";
        case vk::Format::eB8G8R8Sscaled:
            return "VK_FORMAT_B8G8R8_SSCALED";
        case vk::Format::eB8G8R8Uint:
            return "VK_FORMAT_B8G8R8_UINT";
        case vk::Format::eB8G8R8Sint:
            return "VK_FORMAT_B8G8R8_SINT";
        case vk::Format::eB8G8R8Srgb:
            return "VK_FORMAT_B8G8R8_SRGB";
        case vk::Format::eR8G8B8A8Unorm:
            return "VK_FORMAT_R8G8B8A8_UNORM";
        case vk::Format::eR8G8B8A8Snorm:
            return "VK_FORMAT_R8G8B8A8_SNORM";
        case vk::Format::eR8G8B8A8Uscaled:
            return "VK_FORMAT_R8G8B8A8_USCALED";
        case vk::Format::eR8G8B8A8Sscaled:
            return "VK_FORMAT_R8G8B8A8_SSCALED";
        case vk::Format::eR8G8B8A8Uint:
            return "VK_FORMAT_R8G8B8A8_UINT";
        case vk::Format::eR8G8B8A8Sint:
            return "VK_FORMAT_R8G8B8A8_SINT";
        case vk::Format::eR8G8B8A8Srgb:
            return "VK_FORMAT_R8G8B8A8_SRGB";
        case vk::Format::eB8G8R8A8Unorm:
            return "VK_FORMAT_B8G8R8A8_UNORM";
        case vk::Format::eB8G8R8A8Snorm:
            return "VK_FORMAT_B8G8R8A8_SNORM";
        case vk::Format::eB8G8R8A8Uscaled:
            return "VK_FORMAT_B8G8R8A8_USCALED";
        case vk::Format::eB8G8R8A8Sscaled:
            return "VK_FORMAT_B8G8R8A8_SSCALED";
        case vk::Format::eB8G8R8A8Uint:
            return "VK_FORMAT_B8G8R8A8_UINT";
        case vk::Format::eB8G8R8A8Sint:
            return "VK_FORMAT_B8G8R8A8_SINT";
        case vk::Format::eB8G8R8A8Srgb:
            return "VK_FORMAT_B8G8R8A8_SRGB";
        case vk::Format::eA8B8G8R8UnormPack32:
            return "VK_FORMAT_A8B8G8R8_UNORM_PACK32";
        case vk::Format::eA8B8G8R8SnormPack32:
            return "VK_FORMAT_A8B8G8R8_SNORM_PACK32";
        case vk::Format::eA8B8G8R8UscaledPack32:
            return "VK_FORMAT_A8B8G8R8_USCALED_PACK32";
        case vk::Format::eA8B8G8R8SscaledPack32:
            return "VK_FORMAT_A8B8G8R8_SSCALED_PACK32";
        case vk::Format::eA8B8G8R8UintPack32:
            return "VK_FORMAT_A8B8G8R8_UINT_PACK32";
        case vk::Format::eA8B8G8R8SintPack32:
            return "VK_FORMAT_A8B8G8R8_SINT_PACK32";
        case vk::Format::eA8B8G8R8SrgbPack32:
            return "VK_FORMAT_A8B8G8R8_SRGB_PACK32";
        case vk::Format::eA2R10G10B10UnormPack32:
            return "VK_FORMAT_A2R10G10B10_UNORM_PACK32";
        case vk::Format::eA2R10G10B10SnormPack32:
            return "VK_FORMAT_A2R10G10B10_SNORM_PACK32";
        case vk::Format::eA2R10G10B10UscaledPack32:
            return "VK_FORMAT_A2R10G10B10_USCALED_PACK32";
        case vk::Format::eA2R10G10B10SscaledPack32:
            return "VK_FORMAT_A2R10G10B10_SSCALED_PACK32";
        case vk::Format::eA2R10G10B10UintPack32:
            return "VK_FORMAT_A2R10G10B10_UINT_PACK32";
        case vk::Format::eA2R10G10B10SintPack32:
            return "VK_FORMAT_A2R10G10B10_SINT_PACK32";
        case vk::Format::eA2B10G10R10UnormPack32:
            return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
        case vk::Format::eA2B10G10R10SnormPack32:
            return "VK_FORMAT_A2B10G10R10_SNORM_PACK32";
        case vk::Format::eA2B10G10R10UscaledPack32:
            return "VK_FORMAT_A2B10G10R10_USCALED_PACK32";
        case vk::Format::eA2B10G10R10SscaledPack32:
            return "VK_FORMAT_A2B10G10R10_SSCALED_PACK32";
        case vk::Format::eA2B10G10R10UintPack32:
            return "VK_FORMAT_A2B10G10R10_UINT_PACK32";
        case vk::Format::eA2B10G10R10SintPack32:
            return "VK_FORMAT_A2B10G10R10_SINT_PACK32";
        case vk::Format::eR16Unorm:
            return "VK_FORMAT_R16_UNORM";
        case vk::Format::eR16Snorm:
            return "VK_FORMAT_R16_SNORM";
        case vk::Format::eR16Uscaled:
            return "VK_FORMAT_R16_USCALED";
        case vk::Format::eR16Sscaled:
            return "VK_FORMAT_R16_SSCALED";
        case vk::Format::eR16Uint:
            return "VK_FORMAT_R16_UINT";
        case vk::Format::eR16Sint:
            return "VK_FORMAT_R16_SINT";
        case vk::Format::eR16Sfloat:
            return "VK_FORMAT_R16_SFLOAT";
        case vk::Format::eR16G16Unorm:
            return "VK_FORMAT_R16G16_UNORM";
        case vk::Format::eR16G16Snorm:
            return "VK_FORMAT_R16G16_SNORM";
        case vk::Format::eR16G16Uscaled:
            return "VK_FORMAT_R16G16_USCALED";
        case vk::Format::eR16G16Sscaled:
            return "VK_FORMAT_R16G16_SSCALED";
        case vk::Format::eR16G16Uint:
            return "VK_FORMAT_R16G16_UINT";
        case vk::Format::eR16G16Sint:
            return "VK_FORMAT_R16G16_SINT";
        case vk::Format::eR16G16Sfloat:
            return "VK_FORMAT_R16G16_SFLOAT";
        case vk::Format::eR16G16B16Unorm:
            return "VK_FORMAT_R16G16B16_UNORM";
        case vk::Format::eR16G16B16Snorm:
            return "VK_FORMAT_R16G16B16_SNORM";
        case vk::Format::eR16G16B16Uscaled:
            return "VK_FORMAT_R16G16B16_USCALED";
        case vk::Format::eR16G16B16Sscaled:
            return "VK_FORMAT_R16G16B16_SSCALED";
        case vk::Format::eR16G16B16Uint:
            return "VK_FORMAT_R16G16B16_UINT";
        case vk::Format::eR16G16B16Sint:
            return "VK_FORMAT_R16G16B16_SINT";
        case vk::Format::eR16G16B16Sfloat:
            return "VK_FORMAT_R16G16B16_SFLOAT";
        case vk::Format::eR16G16B16A16Unorm:
            return "VK_FORMAT_R16G16B16A16_UNORM";
        case vk::Format::eR16G16B16A16Snorm:
            return "VK_FORMAT_R16G16B16A16_SNORM";
        case vk::Format::eR16G16B16A16Uscaled:
            return "VK_FORMAT_R16G16B16A16_USCALED";
        case vk::Format::eR16G16B16A16Sscaled:
            return "VK_FORMAT_R16G16B16A16_SSCALED";
        case vk::Format::eR16G16B16A16Uint:
            return "VK_FORMAT_R16G16B16A16_UINT";
        case vk::Format::eR16G16B16A16Sint:
            return "VK_FORMAT_R16G16B16A16_SINT";
        case vk::Format::eR16G16B16A16Sfloat:
            return "VK_FORMAT_R16G16B16A16_SFLOAT";
        case vk::Format::eR32Uint:
            return "VK_FORMAT_R32_UINT";
        case vk::Format::eR32Sint:
            return "VK_FORMAT_R32_SINT";
        case vk::Format::eR32Sfloat:
            return "VK_FORMAT_R32_SFLOAT";
        case vk::Format::eR32G32Uint:
            return "VK_FORMAT_R32G32_UINT";
        case vk::Format::eR32G32Sint:
            return "VK_FORMAT_R32G32_SINT";
        case vk::Format::eR32G32Sfloat:
            return "VK_FORMAT_R32G32_SFLOAT";
        case vk::Format::eR32G32B32Uint:
            return "VK_FORMAT_R32G32B32_UINT";
        case vk::Format::eR32G32B32Sint:
            return "VK_FORMAT_R32G32B32_SINT";
        case vk::Format::eR32G32B32Sfloat:
            return "VK_FORMAT_R32G32B32_SFLOAT";
        case vk::Format::eR32G32B32A32Uint:
            return "VK_FORMAT_R32G32B32A32_UINT";
        case vk::Format::eR32G32B32A32Sint:
            return "VK_FORMAT_R32G32B32A32_SINT";
        case vk::Format::eR32G32B32A32Sfloat:
            return "VK_FORMAT_R32G32B32A32_SFLOAT";
        case vk::Format::eR64Uint:
            return "VK_FORMAT_R64_UINT";
        case vk::Format::eR64Sint:
            return "VK_FORMAT_R64_SINT";
        case vk::Format::eR64Sfloat:
            return "VK_FORMAT_R64_SFLOAT";
        case vk::Format::eR64G64Uint:
            return "VK_FORMAT_R64G64_UINT";
        case vk::Format::eR64G64Sint:
            return "VK_FORMAT_R64G64_SINT";
        case vk::Format::eR64G64Sfloat:
            return "VK_FORMAT_R64G64_SFLOAT";
        case vk::Format::eR64G64B64Uint:
            return "VK_FORMAT_R64G64B64_UINT";
        case vk::Format::eR64G64B64Sint:
            return "VK_FORMAT_R64G64B64_SINT";
        case vk::Format::eR64G64B64Sfloat:
            return "VK_FORMAT_R64G64B64_SFLOAT";
        case vk::Format::eR64G64B64A64Uint:
            return "VK_FORMAT_R64G64B64A64_UINT";
        case vk::Format::eR64G64B64A64Sint:
            return "VK_FORMAT_R64G64B64A64_SINT";
        case vk::Format::eR64G64B64A64Sfloat:
            return "VK_FORMAT_R64G64B64A64_SFLOAT";
        case vk::Format::eB10G11R11UfloatPack32:
            return "VK_FORMAT_B10G11R11_UFLOAT_PACK32";
        case vk::Format::eE5B9G9R9UfloatPack32:
            return "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32";
        case vk::Format::eD16Unorm:
            return "VK_FORMAT_D16_UNORM";
        case vk::Format::eX8D24UnormPack32:
            return "VK_FORMAT_X8_D24_UNORM_PACK32";
        case vk::Format::eD32Sfloat:
            return "VK_FORMAT_D32_SFLOAT";
        case vk::Format::eS8Uint:
            return "VK_FORMAT_S8_UINT";
        case vk::Format::eD16UnormS8Uint:
            return "VK_FORMAT_D16_UNORM_S8_UINT";
        case vk::Format::eD24UnormS8Uint:
            return "VK_FORMAT_D24_UNORM_S8_UINT";
        case vk::Format::eD32SfloatS8Uint:
            return "VK_FORMAT_D32_SFLOAT_S8_UINT";
        case vk::Format::eUndefined:
            return "VK_FORMAT_UNDEFINED";
        default:
            return "VK_FORMAT_INVALID";
    }
}

const std::string to_string(vk::PresentModeKHR present_mode) {
    switch (present_mode) {
        case vk::PresentModeKHR::eMailbox:
            return "VK_PRESENT_MODE_MAILBOX_KHR";
        case vk::PresentModeKHR::eImmediate:
            return "VK_PRESENT_MODE_IMMEDIATE_KHR";
        case vk::PresentModeKHR::eFifo:
            return "VK_PRESENT_MODE_FIFO_KHR";
        case vk::PresentModeKHR::eFifoRelaxed:
            return "VK_PRESENT_MODE_FIFO_RELAXED_KHR";
        case vk::PresentModeKHR::eSharedContinuousRefresh:
            return "VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR";
        case vk::PresentModeKHR::eSharedDemandRefresh:
            return "VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR";
        default:
            return "UNKNOWN_PRESENT_MODE";
    }
}

const std::string to_string(vk::Result result) {
    switch (result) {
#define STR(r)             \
    case vk::Result::e##r: \
        return #r
        STR(NotReady);
        STR(Timeout);
        STR(EventSet);
        STR(EventReset);
        STR(Incomplete);
        STR(ErrorOutOfHostMemory);
        STR(ErrorOutOfDeviceMemory);
        STR(ErrorInitializationFailed);
        STR(ErrorDeviceLost);
        STR(ErrorMemoryMapFailed);
        STR(ErrorLayerNotPresent);
        STR(ErrorExtensionNotPresent);
        STR(ErrorFeatureNotPresent);
        STR(ErrorIncompatibleDriver);
        STR(ErrorTooManyObjects);
        STR(ErrorFormatNotSupported);
        STR(ErrorSurfaceLostKHR);
        STR(ErrorNativeWindowInUseKHR);
        STR(SuboptimalKHR);
        STR(ErrorOutOfDateKHR);
        STR(ErrorIncompatibleDisplayKHR);
        STR(ErrorValidationFailedEXT);
        STR(ErrorInvalidShaderNV);
#undef STR
        default:
            return "UNKNOWN_ERROR";
    }
}

const std::string to_string(vk::SurfaceTransformFlagBitsKHR transform_flag) {
    switch (transform_flag) {
        case vk::SurfaceTransformFlagBitsKHR::eIdentity:
            return "SURFACE_TRANSFORM_IDENTITY";
        case vk::SurfaceTransformFlagBitsKHR::eRotate90:
            return "SURFACE_TRANSFORM_ROTATE_90";
        case vk::SurfaceTransformFlagBitsKHR::eRotate180:
            return "SURFACE_TRANSFORM_ROTATE_180";
        case vk::SurfaceTransformFlagBitsKHR::eRotate270:
            return "SURFACE_TRANSFORM_ROTATE_270";
        case vk::SurfaceTransformFlagBitsKHR::eHorizontalMirror:
            return "SURFACE_TRANSFORM_HORIZONTAL_MIRROR";
        case vk::SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90:
            return "SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90";
        case vk::SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180:
            return "SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180";
        case vk::SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270:
            return "SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270";
        case vk::SurfaceTransformFlagBitsKHR::eInherit:
            return "SURFACE_TRANSFORM_INHERIT";
        default:
            return "[Unknown transform flag]";
    }
}

const std::string to_string(vk::SurfaceFormatKHR surface_format) {
    std::string surface_format_string = vox::to_string(surface_format.format) + ", ";

    switch (surface_format.colorSpace) {
        case vk::ColorSpaceKHR::eSrgbNonlinear:
            surface_format_string += "VK_COLORSPACE_SRGB_NONLINEAR_KHR";
            break;
        default:
            surface_format_string += "UNKNOWN COLOR SPACE";
    }
    return surface_format_string;
}

const std::string to_string(vk::CompositeAlphaFlagBitsKHR composite_alpha) {
    switch (composite_alpha) {
        case vk::CompositeAlphaFlagBitsKHR::eOpaque:
            return "VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR";
        case vk::CompositeAlphaFlagBitsKHR::ePreMultiplied:
            return "VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR";
        case vk::CompositeAlphaFlagBitsKHR::ePostMultiplied:
            return "VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR";
        case vk::CompositeAlphaFlagBitsKHR::eInherit:
            return "VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR";
        default:
            return "UNKNOWN COMPOSITE ALPHA FLAG";
    }
}

const std::string to_string(vk::ImageUsageFlagBits image_usage) {
    switch (image_usage) {
        case vk::ImageUsageFlagBits::eTransferSrc:
            return "VK_IMAGE_USAGE_TRANSFER_SRC_BIT";
        case vk::ImageUsageFlagBits::eTransferDst:
            return "VK_IMAGE_USAGE_TRANSFER_DST_BIT";
        case vk::ImageUsageFlagBits::eSampled:
            return "VK_IMAGE_USAGE_SAMPLED_BIT";
        case vk::ImageUsageFlagBits::eStorage:
            return "VK_IMAGE_USAGE_STORAGE_BIT";
        case vk::ImageUsageFlagBits::eColorAttachment:
            return "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT";
        case vk::ImageUsageFlagBits::eDepthStencilAttachment:
            return "VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT";
        case vk::ImageUsageFlagBits::eTransientAttachment:
            return "VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT";
        case vk::ImageUsageFlagBits::eInputAttachment:
            return "VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT";
        default:
            return "UNKNOWN IMAGE USAGE FLAG";
    }
}

std::string to_string(vk::Extent2D extent) {
    return fmt::format("{}x{}", extent.width, extent.height);
}

const std::string to_string(vk::SampleCountFlagBits flags) {
    std::string result{""};
    bool append = false;
    if (flags & vk::SampleCountFlagBits::e1) {
        result += "1";
        append = true;
    }
    if (flags & vk::SampleCountFlagBits::e2) {
        result += append ? "/" : "";
        result += "2";
        append = true;
    }
    if (flags & vk::SampleCountFlagBits::e4) {
        result += append ? "/" : "";
        result += "4";
        append = true;
    }
    if (flags & vk::SampleCountFlagBits::e8) {
        result += append ? "/" : "";
        result += "8";
        append = true;
    }
    if (flags & vk::SampleCountFlagBits::e16) {
        result += append ? "/" : "";
        result += "16";
        append = true;
    }
    if (flags & vk::SampleCountFlagBits::e32) {
        result += append ? "/" : "";
        result += "32";
        append = true;
    }
    if (flags & vk::SampleCountFlagBits::e64) {
        result += append ? "/" : "";
        result += "64";
    }
    return result;
}

const std::string to_string(vk::PhysicalDeviceType type) {
    switch (type) {
        case vk::PhysicalDeviceType::eOther:
            return "VK_PHYSICAL_DEVICE_TYPE_OTHER";
        case vk::PhysicalDeviceType::eIntegratedGpu:
            return "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU";
        case vk::PhysicalDeviceType::eDiscreteGpu:
            return "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU";
        case vk::PhysicalDeviceType::eVirtualGpu:
            return "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU";
        case vk::PhysicalDeviceType::eCpu:
            return "VK_PHYSICAL_DEVICE_TYPE_CPU";
        default:
            return "UNKNOWN_DEVICE_TYPE";
    }
}

const std::string to_string(vk::ImageTiling tiling) {
    switch (tiling) {
        case vk::ImageTiling::eOptimal:
            return "VK_IMAGE_TILING_OPTIMAL";
        case vk::ImageTiling::eLinear:
            return "VK_IMAGE_TILING_LINEAR";
        default:
            return "UNKOWN_TILING_METHOD";
    }
}

const std::string to_string(vk::ImageType type) {
    switch (type) {
        case vk::ImageType::e1D:
            return "VK_IMAGE_TYPE_1D";
        case vk::ImageType::e2D:
            return "VK_IMAGE_TYPE_2D";
        case vk::ImageType::e3D:
            return "VK_IMAGE_TYPE_3D";
        default:
            return "UNKOWN_IMAGE_TYPE";
    }
}

const std::string to_string(vk::BlendFactor blend) {
    switch (blend) {
        case vk::BlendFactor::eZero:
            return "VK_BLEND_FACTOR_ZERO";
        case vk::BlendFactor::eOne:
            return "VK_BLEND_FACTOR_ONE";
        case vk::BlendFactor::eSrcColor:
            return "VK_BLEND_FACTOR_SRC_COLOR";
        case vk::BlendFactor::eOneMinusSrcColor:
            return "VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR";
        case vk::BlendFactor::eDstColor:
            return "VK_BLEND_FACTOR_DST_COLOR";
        case vk::BlendFactor::eOneMinusDstColor:
            return "VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR";
        case vk::BlendFactor::eSrcAlpha:
            return "VK_BLEND_FACTOR_SRC_ALPHA";
        case vk::BlendFactor::eOneMinusSrcAlpha:
            return "VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA";
        case vk::BlendFactor::eDstAlpha:
            return "VK_BLEND_FACTOR_DST_ALPHA";
        case vk::BlendFactor::eOneMinusDstAlpha:
            return "VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA";
        case vk::BlendFactor::eConstantColor:
            return "VK_BLEND_FACTOR_CONSTANT_COLOR";
        case vk::BlendFactor::eOneMinusConstantColor:
            return "VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR";
        case vk::BlendFactor::eConstantAlpha:
            return "VK_BLEND_FACTOR_CONSTANT_ALPHA";
        case vk::BlendFactor::eOneMinusConstantAlpha:
            return "VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA";
        case vk::BlendFactor::eSrcAlphaSaturate:
            return "VK_BLEND_FACTOR_SRC_ALPHA_SATURATE";
        case vk::BlendFactor::eSrc1Color:
            return "VK_BLEND_FACTOR_SRC1_COLOR";
        case vk::BlendFactor::eOneMinusSrc1Color:
            return "VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR";
        case vk::BlendFactor::eSrc1Alpha:
            return "VK_BLEND_FACTOR_SRC1_ALPHA";
        case vk::BlendFactor::eOneMinusSrc1Alpha:
            return "VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA";
        default:
            return "Unknown Blend Factor";
    }
}

const std::string to_string(vk::VertexInputRate rate) {
    switch (rate) {
        case vk::VertexInputRate::eVertex:
            return "VK_VERTEX_INPUT_RATE_VERTEX";
        case vk::VertexInputRate::eInstance:
            return "VK_VERTEX_INPUT_RATE_INSTANCE";
        default:
            return "Unknown Rate";
    }
}

const std::string to_string_vk_bool(vk::Bool32 state) {
    if (state == VK_TRUE) {
        return "true";
    }

    return "false";
}

const std::string to_string(vk::PrimitiveTopology topology) {
    if (topology == vk::PrimitiveTopology::ePointList) {
        return "VK_PRIMITIVE_TOPOLOGY_POINT_LIST";
    }
    if (topology == vk::PrimitiveTopology::eLineList) {
        return "VK_PRIMITIVE_TOPOLOGY_LINE_LIST";
    }
    if (topology == vk::PrimitiveTopology::eLineStrip) {
        return "VK_PRIMITIVE_TOPOLOGY_LINE_STRIP";
    }
    if (topology == vk::PrimitiveTopology::eTriangleList) {
        return "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST";
    }
    if (topology == vk::PrimitiveTopology::eTriangleStrip) {
        return "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP";
    }
    if (topology == vk::PrimitiveTopology::eTriangleFan) {
        return "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN";
    }
    if (topology == vk::PrimitiveTopology::eLineListWithAdjacency) {
        return "VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY";
    }
    if (topology == vk::PrimitiveTopology::eLineStripWithAdjacency) {
        return "VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY";
    }
    if (topology == vk::PrimitiveTopology::eTriangleListWithAdjacency) {
        return "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY";
    }
    if (topology == vk::PrimitiveTopology::eTriangleStripWithAdjacency) {
        return "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY";
    }
    if (topology == vk::PrimitiveTopology::ePatchList) {
        return "VK_PRIMITIVE_TOPOLOGY_PATCH_LIST";
    }

    return "UNKOWN TOPOLOGY";
}

const std::string to_string(vk::FrontFace face) {
    if (face == vk::FrontFace::eCounterClockwise) {
        return "VK_FRONT_FACE_COUNTER_CLOCKWISE";
    }
    if (face == vk::FrontFace::eClockwise) {
        return "VK_FRONT_FACE_CLOCKWISE";
    }
    return "UNKOWN";
}

const std::string to_string(vk::PolygonMode mode) {
    if (mode == vk::PolygonMode::eFill) {
        return "VK_POLYGON_MODE_FILL";
    }
    if (mode == vk::PolygonMode::eLine) {
        return "VK_POLYGON_MODE_LINE";
    }
    if (mode == vk::PolygonMode::ePoint) {
        return "VK_POLYGON_MODE_POINT";
    }
    if (mode == vk::PolygonMode::eFillRectangleNV) {
        return "VK_POLYGON_MODE_FILL_RECTANGLE_NV";
    }
    return "UNKOWN";
}

const std::string to_string(vk::CompareOp operation) {
    if (operation == vk::CompareOp::eNever) {
        return "NEVER";
    }
    if (operation == vk::CompareOp::eLess) {
        return "LESS";
    }
    if (operation == vk::CompareOp::eEqual) {
        return "EQUAL";
    }
    if (operation == vk::CompareOp::eLessOrEqual) {
        return "LESS_OR_EQUAL";
    }
    if (operation == vk::CompareOp::eGreater) {
        return "GREATER";
    }
    if (operation == vk::CompareOp::eNotEqual) {
        return "NOT_EQUAL";
    }
    if (operation == vk::CompareOp::eGreaterOrEqual) {
        return "GREATER_OR_EQUAL";
    }
    if (operation == vk::CompareOp::eAlways) {
        return "ALWAYS";
    }
    return "Unkown";
}

const std::string to_string(vk::StencilOp operation) {
    if (operation == vk::StencilOp::eKeep) {
        return "KEEP";
    }
    if (operation == vk::StencilOp::eZero) {
        return "ZERO";
    }
    if (operation == vk::StencilOp::eReplace) {
        return "REPLACE";
    }
    if (operation == vk::StencilOp::eIncrementAndClamp) {
        return "INCREMENT_AND_CLAMP";
    }
    if (operation == vk::StencilOp::eDecrementAndClamp) {
        return "DECREMENT_AND_CLAMP";
    }
    if (operation == vk::StencilOp::eInvert) {
        return "INVERT";
    }
    if (operation == vk::StencilOp::eIncrementAndWrap) {
        return "INCREMENT_AND_WRAP";
    }
    if (operation == vk::StencilOp::eDecrementAndWrap) {
        return "DECREMENT_AND_WRAP";
    }
    return "Unkown";
}

const std::string to_string(vk::LogicOp operation) {
    if (operation == vk::LogicOp::eClear) {
        return "CLEAR";
    }
    if (operation == vk::LogicOp::eAnd) {
        return "AND";
    }
    if (operation == vk::LogicOp::eAndReverse) {
        return "AND_REVERSE";
    }
    if (operation == vk::LogicOp::eCopy) {
        return "COPY";
    }
    if (operation == vk::LogicOp::eAndInverted) {
        return "AND_INVERTED";
    }
    if (operation == vk::LogicOp::eNoOp) {
        return "NO_OP";
    }
    if (operation == vk::LogicOp::eXor) {
        return "XOR";
    }
    if (operation == vk::LogicOp::eOr) {
        return "OR";
    }
    if (operation == vk::LogicOp::eNor) {
        return "NOR";
    }
    if (operation == vk::LogicOp::eEquivalent) {
        return "EQUIVALENT";
    }
    if (operation == vk::LogicOp::eInvert) {
        return "INVERT";
    }
    if (operation == vk::LogicOp::eOrReverse) {
        return "OR_REVERSE";
    }
    if (operation == vk::LogicOp::eCopyInverted) {
        return "COPY_INVERTED";
    }
    if (operation == vk::LogicOp::eOrInverted) {
        return "OR_INVERTED";
    }
    if (operation == vk::LogicOp::eNand) {
        return "NAND";
    }
    if (operation == vk::LogicOp::eSet) {
        return "SET";
    }
    return "Unkown";
}

const std::string to_string(vk::BlendOp operation) {
    if (operation == vk::BlendOp::eAdd) {
        return "ADD";
    }

    if (operation == vk::BlendOp::eSubtract) {
        return "SUBTRACT";
    }

    if (operation == vk::BlendOp::eReverseSubtract) {
        return "REVERSE_SUBTRACT";
    }

    if (operation == vk::BlendOp::eMin) {
        return "MIN";
    }

    if (operation == vk::BlendOp::eMax) {
        return "MAX";
    }

    if (operation == vk::BlendOp::eZeroEXT) {
        return "ZERO_EXT";
    }

    if (operation == vk::BlendOp::eSrcEXT) {
        return "SRC_EXT";
    }

    if (operation == vk::BlendOp::eDstEXT) {
        return "DST_EXT";
    }

    if (operation == vk::BlendOp::eSrcOverEXT) {
        return "SRC_OVER_EXT";
    }

    if (operation == vk::BlendOp::eDstOverEXT) {
        return "DST_OVER_EXT";
    }

    if (operation == vk::BlendOp::eSrcInEXT) {
        return "SRC_IN_EXT";
    }

    if (operation == vk::BlendOp::eDstInEXT) {
        return "DST_IN_EXT";
    }

    if (operation == vk::BlendOp::eSrcOutEXT) {
        return "SRC_OUT_EXT";
    }

    if (operation == vk::BlendOp::eDstOutEXT) {
        return "DST_OUT_EXT";
    }

    if (operation == vk::BlendOp::eSrcAtopEXT) {
        return "SRC_ATOP_EXT";
    }

    if (operation == vk::BlendOp::eDstAtopEXT) {
        return "DST_ATOP_EXT";
    }

    if (operation == vk::BlendOp::eXorEXT) {
        return "XOR_EXT";
    }

    if (operation == vk::BlendOp::eMultiplyEXT) {
        return "MULTIPLY_EXT";
    }

    if (operation == vk::BlendOp::eScreenEXT) {
        return "SCREEN_EXT";
    }

    if (operation == vk::BlendOp::eOverlayEXT) {
        return "OVERLAY_EXT";
    }

    if (operation == vk::BlendOp::eDarkenEXT) {
        return "DARKEN_EXT";
    }

    if (operation == vk::BlendOp::eLightenEXT) {
        return "LIGHTEN_EXT";
    }

    if (operation == vk::BlendOp::eColordodgeEXT) {
        return "COLORDODGE_EXT";
    }

    if (operation == vk::BlendOp::eColorburnEXT) {
        return "COLORBURN_EXT";
    }

    if (operation == vk::BlendOp::eHardlightEXT) {
        return "HARDLIGHT_EXT";
    }

    if (operation == vk::BlendOp::eSoftlightEXT) {
        return "SOFTLIGHT_EXT";
    }

    if (operation == vk::BlendOp::eDifferenceEXT) {
        return "DIFFERENCE_EXT";
    }

    if (operation == vk::BlendOp::eExclusionEXT) {
        return "EXCLUSION_EXT";
    }

    if (operation == vk::BlendOp::eInvertEXT) {
        return "INVERT_EXT";
    }

    if (operation == vk::BlendOp::eInvertRgbEXT) {
        return "INVERT_RGB_EXT";
    }

    if (operation == vk::BlendOp::eLineardodgeEXT) {
        return "LINEARDODGE_EXT";
    }

    if (operation == vk::BlendOp::eLinearburnEXT) {
        return "LINEARBURN_EXT";
    }

    if (operation == vk::BlendOp::eVividlightEXT) {
        return "VIVIDLIGHT_EXT";
    }

    if (operation == vk::BlendOp::eLinearlightEXT) {
        return "LINEARLIGHT_EXT";
    }

    if (operation == vk::BlendOp::ePinlightEXT) {
        return "PINLIGHT_EXT";
    }

    if (operation == vk::BlendOp::eHardmixEXT) {
        return "HARDMIX_EXT";
    }

    if (operation == vk::BlendOp::eHslHueEXT) {
        return "HSL_HUE_EXT";
    }

    if (operation == vk::BlendOp::eHslSaturationEXT) {
        return "HSL_SATURATION_EXT";
    }

    if (operation == vk::BlendOp::eHslColorEXT) {
        return "HSL_COLOR_EXT";
    }

    if (operation == vk::BlendOp::eHslLuminosityEXT) {
        return "HSL_LUMINOSITY_EXT";
    }

    if (operation == vk::BlendOp::ePlusEXT) {
        return "PLUS_EXT";
    }

    if (operation == vk::BlendOp::ePlusClampedEXT) {
        return "PLUS_CLAMPED_EXT";
    }

    if (operation == vk::BlendOp::ePlusClampedAlphaEXT) {
        return "PLUS_CLAMPED_ALPHA_EXT";
    }

    if (operation == vk::BlendOp::ePlusDarkerEXT) {
        return "PLUS_DARKER_EXT";
    }

    if (operation == vk::BlendOp::eMinusEXT) {
        return "MINUS_EXT";
    }

    if (operation == vk::BlendOp::eMinusClampedEXT) {
        return "MINUS_CLAMPED_EXT";
    }

    if (operation == vk::BlendOp::eContrastEXT) {
        return "CONTRAST_EXT";
    }

    if (operation == vk::BlendOp::eInvertOvgEXT) {
        return "INVERT_OVG_EXT";
    }

    if (operation == vk::BlendOp::eRedEXT) {
        return "RED_EXT";
    }

    if (operation == vk::BlendOp::eGreenEXT) {
        return "GREEN_EXT";
    }

    if (operation == vk::BlendOp::eBlueEXT) {
        return "BLUE_EXT";
    }
    return "Unkown";
}

//const std::string to_string(sg::AlphaMode mode) {
//    if (mode == sg::AlphaMode::Blend) {
//        return "Blend";
//    } else if (mode == sg::AlphaMode::Mask) {
//        return "Mask";
//    } else if (mode == sg::AlphaMode::Opaque) {
//        return "Opaque";
//    }
//    return "Unkown";
//}

const std::string to_string(bool flag) {
    if (flag == true) {
        return "true";
    }
    return "false";
}

//const std::string to_string(ShaderResourceType type) {
//    switch (type) {
//        case ShaderResourceType::Input:
//            return "Input";
//        case ShaderResourceType::InputAttachment:
//            return "InputAttachment";
//        case ShaderResourceType::Output:
//            return "Output";
//        case ShaderResourceType::Image:
//            return "Image";
//        case ShaderResourceType::ImageSampler:
//            return "ImageSampler";
//        case ShaderResourceType::ImageStorage:
//            return "ImageStorage";
//        case ShaderResourceType::Sampler:
//            return "Sampler";
//        case ShaderResourceType::BufferUniform:
//            return "BufferUniform";
//        case ShaderResourceType::BufferStorage:
//            return "BufferStorage";
//        case ShaderResourceType::PushConstant:
//            return "PushConstant";
//        case ShaderResourceType::SpecializationConstant:
//            return "SpecializationConstant";
//        default:
//            return "Unkown Type";
//    }
//}

const std::string buffer_usage_to_string(vk::BufferUsageFlags flags) {
    return to_string<vk::BufferUsageFlags, vk::BufferUsageFlagBits>(flags,
                                                                    {{vk::BufferUsageFlagBits::eTransferSrc, "VK_BUFFER_USAGE_TRANSFER_SRC_BIT"},
                                                                     {vk::BufferUsageFlagBits::eTransferDst, "VK_BUFFER_USAGE_TRANSFER_DST_BIT"},
                                                                     {vk::BufferUsageFlagBits::eUniformTexelBuffer, "VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eStorageTexelBuffer, "VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eUniformBuffer, "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eStorageBuffer, "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eIndexBuffer, "VK_BUFFER_USAGE_INDEX_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eVertexBuffer, "VK_BUFFER_USAGE_VERTEX_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eIndirectBuffer, "VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT"},
                                                                     {vk::BufferUsageFlagBits::eShaderDeviceAddress, "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT"},
                                                                     {vk::BufferUsageFlagBits::eTransformFeedbackBufferEXT, "VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT"},
                                                                     {vk::BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT, "VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT"},
                                                                     {vk::BufferUsageFlagBits::eConditionalRenderingEXT, "VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT"},
                                                                     {vk::BufferUsageFlagBits::eRayTracingNV, "VK_BUFFER_USAGE_RAY_TRACING_BIT_NV"},
                                                                     {vk::BufferUsageFlagBits::eShaderDeviceAddressEXT, "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT"},
                                                                     {vk::BufferUsageFlagBits::eShaderDeviceAddressKHR, "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR"}});
}

const std::string shader_stage_to_string(vk::ShaderStageFlags flags) {
    return to_string<vk::ShaderStageFlags, vk::ShaderStageFlagBits>(flags,
                                                                    {{vk::ShaderStageFlagBits::eTessellationControl, "TESSELLATION_CONTROL"},
                                                                     {vk::ShaderStageFlagBits::eTessellationEvaluation, "TESSELLATION_EVALUATION"},
                                                                     {vk::ShaderStageFlagBits::eGeometry, "GEOMETRY"},
                                                                     {vk::ShaderStageFlagBits::eVertex, "VERTEX"},
                                                                     {vk::ShaderStageFlagBits::eFragment, "FRAGMENT"},
                                                                     {vk::ShaderStageFlagBits::eCompute, "COMPUTE"},
                                                                     {vk::ShaderStageFlagBits::eAllGraphics, "ALL GRAPHICS"}});
}

const std::string image_usage_to_string(vk::ImageUsageFlags flags) {
    return to_string<vk::ImageUsageFlags, vk::ImageUsageFlagBits>(flags,
                                                                  {{vk::ImageUsageFlagBits::eTransferSrc, "VK_IMAGE_USAGE_TRANSFER_SRC_BIT"},
                                                                   {vk::ImageUsageFlagBits::eTransferDst, "VK_IMAGE_USAGE_TRANSFER_DST_BIT"},
                                                                   {vk::ImageUsageFlagBits::eSampled, "VK_IMAGE_USAGE_SAMPLED_BIT"},
                                                                   {vk::ImageUsageFlagBits::eStorage, "VK_IMAGE_USAGE_STORAGE_BIT"},
                                                                   {vk::ImageUsageFlagBits::eColorAttachment, "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT"},
                                                                   {vk::ImageUsageFlagBits::eDepthStencilAttachment, "VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT"},
                                                                   {vk::ImageUsageFlagBits::eTransientAttachment, "VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT"},
                                                                   {vk::ImageUsageFlagBits::eInputAttachment, "VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT"}});
}

const std::string image_aspect_to_string(vk::ImageAspectFlags flags) {
    return to_string<vk::ImageAspectFlags, vk::ImageAspectFlagBits>(flags,
                                                                    {{vk::ImageAspectFlagBits::eColor, "VK_IMAGE_ASPECT_COLOR_BIT"},
                                                                     {vk::ImageAspectFlagBits::eDepth, "VK_IMAGE_ASPECT_DEPTH_BIT"},
                                                                     {vk::ImageAspectFlagBits::eStencil, "VK_IMAGE_ASPECT_STENCIL_BIT"},
                                                                     {vk::ImageAspectFlagBits::eMetadata, "VK_IMAGE_ASPECT_METADATA_BIT"},
                                                                     {vk::ImageAspectFlagBits::ePlane0, "VK_IMAGE_ASPECT_PLANE_0_BIT"},
                                                                     {vk::ImageAspectFlagBits::ePlane1, "VK_IMAGE_ASPECT_PLANE_1_BIT"},
                                                                     {vk::ImageAspectFlagBits::ePlane2, "VK_IMAGE_ASPECT_PLANE_2_BIT"}});
}

const std::string cull_mode_to_string(vk::CullModeFlags flags) {
    return to_string<vk::CullModeFlags, vk::CullModeFlagBits>(flags,
                                                              {{vk::CullModeFlagBits::eNone, "VK_CULL_MODE_NONE"},
                                                               {vk::CullModeFlagBits::eFront, "VK_CULL_MODE_FRONT_BIT"},
                                                               {vk::CullModeFlagBits::eBack, "VK_CULL_MODE_BACK_BIT"},
                                                               {vk::CullModeFlagBits::eFrontAndBack, "VK_CULL_MODE_FRONT_AND_BACK"}});
}

const std::string color_component_to_string(vk::ColorComponentFlags flags) {
    return to_string<vk::ColorComponentFlags, vk::ColorComponentFlagBits>(flags,
                                                                          {{vk::ColorComponentFlagBits::eR, "R"},
                                                                           {vk::ColorComponentFlagBits::eG, "G"},
                                                                           {vk::ColorComponentFlagBits::eB, "B"},
                                                                           {vk::ColorComponentFlagBits::eA, "A"}});
}

std::vector<std::string> split(const std::string &input, char delim) {
    std::vector<std::string> tokens;

    std::stringstream sstream(input);
    std::string token;
    while (std::getline(sstream, token, delim)) {
        tokens.push_back(token);
    }

    return tokens;
}
}// namespace vox
