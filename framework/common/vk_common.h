/* Copyright (c) 2018-2021, Arm Limited and Contributors
 * Copyright (c) 2019-2021, Sascha Willems
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

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <vk_mem_alloc.h>
#include <volk.h>
#include <vulkan/vulkan.hpp>
#include "common/logging.h"

#define VK_FLAGS_NONE 0// Custom define for better code readability

#define DEFAULT_FENCE_TIMEOUT 100000000000// Default fence timeout in nanoseconds

template<class T>
using ShaderStageMap = std::map<vk::ShaderStageFlagBits, T>;

template<class T>
using BindingMap = std::map<uint32_t, std::map<uint32_t, T>>;

namespace vox {
/**
 * @brief Helper function to determine if a Vulkan format is depth only.
 * @param format Vulkan format to check.
 * @return True if format is a depth only, false otherwise.
 */
bool is_depth_only_format(vk::Format format);

/**
 * @brief Helper function to determine if a Vulkan format is depth or stencil.
 * @param format Vulkan format to check.
 * @return True if format is a depth or stencil, false otherwise.
 */
bool is_depth_stencil_format(vk::Format format);

/**
 * @brief Helper function to determine a suitable supported depth format based on a priority list
 * @param physical_device The physical device to check the depth formats against
 * @param depth_only (Optional) Wether to include the stencil component in the format or not
 * @param depth_format_priority_list (Optional) The list of depth formats to prefer over one another
 *		  By default we start with the highest precision packed format
 * @return The valid suited depth format
 */
vk::Format get_suitable_depth_format(vk::PhysicalDevice physical_device,
                                     bool depth_only = false,
                                     const std::vector<vk::Format> &depth_format_priority_list = {
                                         vk::Format::eD32Sfloat,
                                         vk::Format::eD24UnormS8Uint,
                                         vk::Format::eD16Unorm});

/**
 * @brief Helper function to determine if a Vulkan descriptor type is a dynamic storage buffer or dynamic uniform buffer.
 * @param descriptor_type Vulkan descriptor type to check.
 * @return True if type is dynamic buffer, false otherwise.
 */
bool is_dynamic_buffer_descriptor_type(vk::DescriptorType descriptor_type);

/**
 * @brief Helper function to determine if a Vulkan descriptor type is a buffer (either uniform or storage buffer, dynamic or not).
 * @param descriptor_type Vulkan descriptor type to check.
 * @return True if type is buffer, false otherwise.
 */
bool is_buffer_descriptor_type(vk::DescriptorType descriptor_type);

/**
 * @brief Helper function to get the bits per pixel of a Vulkan format.
 * @param format Vulkan format to check.
 * @return The bits per pixel of the given format, -1 for invalid formats.
 */
int32_t get_bits_per_pixel(vk::Format format);

/**
 * @brief Helper function to create a vk::ShaderModule
 * @param filename The shader location
 * @param device The logical device
 * @param stage The shader stage
 * @return The string to return.
 */
vk::ShaderModule load_shader(const std::string &filename, vk::Device device, vk::ShaderStageFlagBits stage);

/**
 * @brief Image memory barrier structure used to define
 *        memory access for an image view during command recording.
 */
struct ImageMemoryBarrier {
    vk::PipelineStageFlags src_stage_mask{vk::PipelineStageFlagBits::eBottomOfPipe};

    vk::PipelineStageFlags dst_stage_mask{vk::PipelineStageFlagBits::eTopOfPipe};

    vk::AccessFlags src_access_mask{0};

    vk::AccessFlags dst_access_mask{0};

    vk::ImageLayout old_layout{vk::ImageLayout::eUndefined};

    vk::ImageLayout new_layout{vk::ImageLayout::eUndefined};

    uint32_t old_queue_family{VK_QUEUE_FAMILY_IGNORED};

    uint32_t new_queue_family{VK_QUEUE_FAMILY_IGNORED};
};

/**
* @brief Buffer memory barrier structure used to define
*        memory access for a buffer during command recording.
*/
struct BufferMemoryBarrier {
    vk::PipelineStageFlags src_stage_mask{vk::PipelineStageFlagBits::eBottomOfPipe};

    vk::PipelineStageFlags dst_stage_mask{vk::PipelineStageFlagBits::eTopOfPipe};

    vk::AccessFlags src_access_mask{0};

    vk::AccessFlags dst_access_mask{0};
};

/**
* @brief Put an image memory barrier for setting an image layout on the sub resource into the given command buffer
*/
void set_image_layout(
    vk::CommandBuffer command_buffer,
    vk::Image image,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::ImageSubresourceRange subresource_range,
    vk::PipelineStageFlags src_mask = vk::PipelineStageFlagBits::eAllCommands,
    vk::PipelineStageFlags dst_mask = vk::PipelineStageFlagBits::eAllCommands);

/**
* @brief Uses a fixed sub resource layout with first mip level and layer
*/
void set_image_layout(
    vk::CommandBuffer command_buffer,
    vk::Image image,
    vk::ImageAspectFlags aspect_mask,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::PipelineStageFlags src_mask = vk::PipelineStageFlagBits::eAllCommands,
    vk::PipelineStageFlags dst_mask = vk::PipelineStageFlagBits::eAllCommands);

/**
* @brief Insert an image memory barrier into the command buffer
*/
void insert_image_memory_barrier(
    vk::CommandBuffer command_buffer,
    vk::Image image,
    vk::AccessFlags src_access_mask,
    vk::AccessFlags dst_access_mask,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::PipelineStageFlags src_stage_mask,
    vk::PipelineStageFlags dst_stage_mask,
    vk::ImageSubresourceRange subresource_range);

/**
 * @brief Load and store info for a render pass attachment.
 */
struct LoadStoreInfo {
    vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eClear;

    vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore;
};

namespace gbuffer {
/**
  * @return Load store info to load all and store only the swapchain
  */
std::vector<LoadStoreInfo> get_load_all_store_swapchain();

/**
  * @return Load store info to clear all and store only the swapchain
  */
std::vector<LoadStoreInfo> get_clear_all_store_swapchain();

/**
  * @return Load store info to clear and store all images
  */
std::vector<LoadStoreInfo> get_clear_store_all();

/**
  * @return Default clear values for the G-buffer
  */
std::vector<vk::ClearValue> get_clear_value();
}// namespace gbuffer

// helper functions not backed by vk_common.h
inline vk::CommandBuffer allocate_command_buffer(vk::Device device, vk::CommandPool command_pool,
                                                 vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) {
    vk::CommandBufferAllocateInfo command_buffer_allocate_info(command_pool, level, 1);
    return device.allocateCommandBuffers(command_buffer_allocate_info).front();
}

inline vk::DescriptorSet allocate_descriptor_set(vk::Device device, vk::DescriptorPool descriptor_pool,
                                                 vk::DescriptorSetLayout descriptor_set_layout) {
#if defined(ANDROID)
    vk::DescriptorSetAllocateInfo descriptor_set_allocate_info(descriptor_pool, 1, &descriptor_set_layout);
#else
    vk::DescriptorSetAllocateInfo descriptor_set_allocate_info(descriptor_pool, descriptor_set_layout);
#endif
    return device.allocateDescriptorSets(descriptor_set_allocate_info).front();
}

inline vk::Framebuffer create_framebuffer(vk::Device device, vk::RenderPass render_pass,
                                          std::vector<vk::ImageView> const &attachments, vk::Extent2D const &extent) {
    vk::FramebufferCreateInfo framebuffer_create_info({}, render_pass, attachments, extent.width, extent.height, 1);
    return device.createFramebuffer(framebuffer_create_info);
}

inline vk::Pipeline create_graphics_pipeline(vk::Device device,
                                             vk::PipelineCache pipeline_cache,
                                             std::array<vk::PipelineShaderStageCreateInfo, 2> const &shader_stages,
                                             vk::PipelineVertexInputStateCreateInfo const &vertex_input_state,
                                             vk::PrimitiveTopology primitive_topology,
                                             vk::CullModeFlags cull_mode,
                                             vk::FrontFace front_face,
                                             std::vector<vk::PipelineColorBlendAttachmentState> const &blend_attachment_states,
                                             vk::PipelineDepthStencilStateCreateInfo const &depth_stencil_state,
                                             vk::PipelineLayout pipeline_layout,
                                             vk::RenderPass render_pass) {
    vk::PipelineInputAssemblyStateCreateInfo input_assembly_state({}, primitive_topology, false);

    vk::PipelineViewportStateCreateInfo viewport_state({}, 1, nullptr, 1, nullptr);

    vk::PipelineRasterizationStateCreateInfo rasterization_state;
    rasterization_state.polygonMode = vk::PolygonMode::eFill;
    rasterization_state.cullMode = cull_mode;
    rasterization_state.frontFace = front_face;
    rasterization_state.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisample_state({}, vk::SampleCountFlagBits::e1);

    vk::PipelineColorBlendStateCreateInfo color_blend_state({}, false, {}, blend_attachment_states);

    std::array<vk::DynamicState, 2> dynamic_state_enables = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamic_state({}, dynamic_state_enables);

    // Final fullscreen composition pass pipeline
    vk::GraphicsPipelineCreateInfo pipeline_create_info({},
                                                        shader_stages,
                                                        &vertex_input_state,
                                                        &input_assembly_state,
                                                        {},
                                                        &viewport_state,
                                                        &rasterization_state,
                                                        &multisample_state,
                                                        &depth_stencil_state,
                                                        &color_blend_state,
                                                        &dynamic_state,
                                                        pipeline_layout,
                                                        render_pass,
                                                        {},
                                                        {},
                                                        -1);

    vk::Result result;
    vk::Pipeline pipeline;
    std::tie(result, pipeline) = device.createGraphicsPipeline(pipeline_cache, pipeline_create_info);
    assert(result == vk::Result::eSuccess);
    return pipeline;
}

inline vk::ImageAspectFlags get_image_aspect_flags(vk::ImageUsageFlagBits usage, vk::Format format) {
    vk::ImageAspectFlags image_aspect_flags;

    switch (usage) {
        case vk::ImageUsageFlagBits::eColorAttachment:
            image_aspect_flags = vk::ImageAspectFlagBits::eColor;
            break;
        case vk::ImageUsageFlagBits::eDepthStencilAttachment:
            image_aspect_flags = vk::ImageAspectFlagBits::eDepth;
            // Stencil aspect should only be set on depth + stencil formats
            if (vox::is_depth_stencil_format(format) && !vox::is_depth_only_format(format)) {
                image_aspect_flags |= vk::ImageAspectFlagBits::eStencil;
            }
            break;
        default:
            assert(false);
    }

    return image_aspect_flags;
}

inline void submit_and_wait(vk::Device device, vk::Queue queue, std::vector<vk::CommandBuffer> command_buffers,
                            std::vector<vk::Semaphore> semaphores = {}) {
    // Submit command_buffer
    vk::SubmitInfo submit_info(nullptr, {}, command_buffers, semaphores);

    // Create fence to ensure that command_buffer has finished executing
    vk::Fence fence = device.createFence({});

    // Submit to the queue
    queue.submit(submit_info, fence);

    // Wait for the fence to signal that command_buffer has finished executing
    vk::Result result = device.waitForFences(fence, true, DEFAULT_FENCE_TIMEOUT);
    if (result != vk::Result::eSuccess) {
        LOGE("Vulkan error on waitForFences: {}", vk::to_string(result));
        abort();
    }

    // Destroy the fence
    device.destroyFence(fence);
}

}// namespace vox
