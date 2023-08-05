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

#pragma once

#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <volk.h>

namespace vox {
enum class ShaderResourceType;

namespace sg {
enum class AlphaMode;
}// namespace sg

std::vector<std::string> split(const std::string &str, const std::string &delimiter);
std::string join(const std::vector<std::string> &str, const std::string &separator);

/**
 * @brief Helper function to convert a vk::Format enum to a string
 * @param format Vulkan format to convert.
 * @return The string to return.
 */
const std::string to_string(vk::Format format);

/**
 * @brief Helper function to convert a vk::PresentModeKHR to a string
 * @param present_mode Vulkan present mode to convert.
 * @return The string to return.
 */
const std::string to_string(vk::PresentModeKHR present_mode);

/**
 * @brief Helper function to convert a vk::Result enum to a string
 * @param result Vulkan result to convert.
 * @return The string to return.
 */
const std::string to_string(vk::Result result);

/**
 * @brief Helper function to convert a vk::PhysicalDeviceType enum to a string
 * @param type Vulkan physical device type to convert.
 * @return The string to return.
 */
const std::string to_string(vk::PhysicalDeviceType type);

/**
 * @brief Helper function to convert a vk::SurfaceTransformFlagBitsKHR flag to a string
 * @param transform_flag Vulkan surface transform flag bit to convert.
 * @return The string to return.
 */
const std::string to_string(vk::SurfaceTransformFlagBitsKHR transform_flag);

/**
 * @brief Helper function to convert a vk::SurfaceFormatKHR format to a string
 * @param surface_format Vulkan surface format to convert.
 * @return The string to return.
 */
const std::string to_string(vk::SurfaceFormatKHR surface_format);

/**
 * @brief Helper function to convert a vk::CompositeAlphaFlagBitsKHR flag to a string
 * @param composite_alpha Vulkan composite alpha flag bit to convert.
 * @return The string to return.
 */
const std::string to_string(vk::CompositeAlphaFlagBitsKHR composite_alpha);

/**
 * @brief Helper function to convert a vk::ImageUsageFlagBits flag to a string
 * @param image_usage Vulkan image usage flag bit to convert.
 * @return The string to return.
 */
const std::string to_string(vk::ImageUsageFlagBits image_usage);

/**
 * @brief Helper function to convert a vk::Extent2D flag to a string
 * @param format Vulkan format to convert.
 * @return The string to return.
 */
std::string to_string(vk::Extent2D format);

/**
 * @brief Helper function to convert vk::SampleCountFlagBits to a string
 * @param flags Vulkan sample count flags to convert
 * @return const std::string 
 */
const std::string to_string(vk::SampleCountFlagBits flags);

/**
 * @brief Helper function to convert vk::ImageTiling to a string 
 * @param tiling Vulkan vk::ImageTiling to convert
 * @return The string to return 
 */
const std::string to_string(vk::ImageTiling tiling);

/**
 * @brief Helper function to convert vk::ImageType to a string 
 * @param type Vulkan vk::ImageType to convert
 * @return The string to return 
 */
const std::string to_string(vk::ImageType type);

/**
 * @brief Helper function to convert vk::BlendFactor to a string 
 * @param blend Vulkan vk::BlendFactor to convert
 * @return The string to return 
 */
const std::string to_string(vk::BlendFactor blend);

/**
 * @brief Helper function to convert vk::VertexInputRate to a string 
 * @param rate Vulkan vk::VertexInputRate to convert
 * @return The string to return 
 */
const std::string to_string(vk::VertexInputRate rate);

/**
 * @brief Helper function to convert vk::Bool32 to a string 
 * @param state Vulkan vk::Bool32 to convert
 * @return The string to return 
 */
const std::string to_string_vk_bool(vk::Bool32 state);

/**
 * @brief Helper function to convert vk::PrimitiveTopology to a string 
 * @param topology Vulkan vk::PrimitiveTopology to convert
 * @return The string to return 
 */
const std::string to_string(vk::PrimitiveTopology topology);

/**
 * @brief Helper function to convert vk::FrontFace to a string 
 * @param face Vulkan vk::FrontFace to convert
 * @return The string to return 
 */
const std::string to_string(vk::FrontFace face);

/**
 * @brief Helper function to convert vk::PolygonMode to a string 
 * @param mode Vulkan vk::PolygonMode to convert
 * @return The string to return 
 */
const std::string to_string(vk::PolygonMode mode);

/**
 * @brief Helper function to convert vk::CompareOp to a string 
 * @param operation Vulkan vk::CompareOp to convert
 * @return The string to return 
 */
const std::string to_string(vk::CompareOp operation);

/**
 * @brief Helper function to convert vk::StencilOp to a string 
 * @param operation Vulkan vk::StencilOp to convert
 * @return The string to return 
 */
const std::string to_string(vk::StencilOp operation);

/**
 * @brief Helper function to convert vk::LogicOp to a string 
 * @param operation Vulkan vk::LogicOp to convert
 * @return The string to return 
 */
const std::string to_string(vk::LogicOp operation);

/**
 * @brief Helper function to convert vk::BlendOp to a string 
 * @param operation Vulkan vk::BlendOp to convert
 * @return The string to return 
 */
const std::string to_string(vk::BlendOp operation);

/**
 * @brief Helper function to convert AlphaMode to a string 
 * @param mode Vulkan AlphaMode to convert
 * @return The string to return 
 */
const std::string to_string(sg::AlphaMode mode);

/**
 * @brief Helper function to convert bool to a string 
 * @param flag Vulkan bool to convert (true/false)
 * @return The string to return 
 */
const std::string to_string(bool flag);

/**
 * @brief Helper function to convert ShaderResourceType to a string 
 * @param type Vulkan ShaderResourceType to convert
 * @return The string to return 
 */
const std::string to_string(ShaderResourceType type);

/**
 * @brief Helper generic function to convert a bitmask to a string of its components
 * @param bitmask The bitmask to convert
 * @param string_map A map of bitmask bits to the string that describe the Vulkan flag
 * @returns A string of the enabled bits in the bitmask
 */
template<typename U, typename T>
inline const std::string to_string(U bitmask, const std::map<T, const char *> string_map) {
    std::stringstream result;
    bool append = false;
    for (const auto &s : string_map) {
        if (bitmask & s.first) {
            if (append) {
                result << " / ";
            }
            result << s.second;
            append = true;
        }
    }
    return result.str();
}

/**
 * @brief Helper function to convert vk::BufferUsageFlags to a string
 * @param bitmask The buffer usage bitmask to convert to strings
 * @return The converted string to return
 */
const std::string buffer_usage_to_string(vk::BufferUsageFlags bitmask);

/**
 * @brief Helper function to convert vk::ShaderStageFlags to a string
 * @param bitmask The shader stage bitmask to convert
 * @return The converted string to return
 */
const std::string shader_stage_to_string(vk::ShaderStageFlags bitmask);

/**
 * @brief Helper function to convert vk::ImageUsageFlags to a string
 * @param bitmask The image usage bitmask to convert
 * @return The converted string to return
 */
const std::string image_usage_to_string(vk::ImageUsageFlags bitmask);

/**
 * @brief Helper function to convert vk::ImageAspectFlags to a string
 * @param bitmask The image aspect bitmask to convert
 * @return The converted string to return
 */
const std::string image_aspect_to_string(vk::ImageAspectFlags bitmask);

/**
 * @brief Helper function to convert vk::CullModeFlags to a string
 * @param bitmask The cull mode bitmask to convert
 * @return The converted string to return
 */
const std::string cull_mode_to_string(vk::CullModeFlags bitmask);

/**
 * @brief Helper function to convert vk::ColorComponentFlags to a string
 * @param bitmask The color component bitmask to convert
 * @return The converted string to return
 */
const std::string color_component_to_string(vk::ColorComponentFlags bitmask);

/**
 * @brief Helper function to split a single string into a vector of strings by a delimiter
 * @param input The input string to be split
 * @param delim The character to delimit by
 * @return The vector of tokenized strings
 */
std::vector<std::string> split(const std::string &input, char delim);
}// namespace vox
