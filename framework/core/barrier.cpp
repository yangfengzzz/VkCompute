//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "barrier.h"

#include "device.h"

#define THSVS_SIMPLER_VULKAN_SYNCHRONIZATION_IMPLEMENTATION
#include <thsvs_simpler_vulkan_synchronization.h>

namespace vox::core {
void record_image_barrier(Device &device, VkCommandBuffer cb, const ImageBarrier &barrier) {
    auto range = VkImageSubresourceRange{
        .aspectMask = barrier.aspect_mask,
        .baseMipLevel = 0,
        .levelCount = 0,
        .baseArrayLayer = 0,
        .layerCount = 0,
    };

    ThsvsImageBarrier imageBarrier{
        .prevAccessCount = 1,
        .pPrevAccesses = &barrier.prev_access,
        .nextAccessCount = 1,
        .pNextAccesses = &barrier.next_access,
        .prevLayout = THSVS_IMAGE_LAYOUT_OPTIMAL,
        .nextLayout = THSVS_IMAGE_LAYOUT_OPTIMAL,
        .discardContents = barrier.discard,
        .srcQueueFamilyIndex = device.get_queue_family_index(VK_QUEUE_GRAPHICS_BIT),
        .dstQueueFamilyIndex = device.get_queue_family_index(VK_QUEUE_GRAPHICS_BIT),
        .image = barrier.image,
        .subresourceRange = range,
    };
    thsvsCmdPipelineBarrier(cb, nullptr,
                            0, nullptr,
                            1, &imageBarrier);
}

AccessInfo get_access_info(ThsvsAccessType access_type) {
    switch (access_type) {
        case THSVS_ACCESS_NONE:
            return AccessInfo{
                .stage_mask = 0,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_COMMAND_BUFFER_READ_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
                .access_mask = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_INDIRECT_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                .access_mask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_INDEX_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                .access_mask = VK_ACCESS_INDEX_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_VERTEX_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                .access_mask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_VERTEX_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_VERTEX_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_VERTEX_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TESSELLATION_CONTROL_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_TESSELLATION_CONTROL_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_TESSELLATION_CONTROL_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TESSELLATION_EVALUATION_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_TESSELLATION_EVALUATION_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_TESSELLATION_EVALUATION_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_GEOMETRY_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_GEOMETRY_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_GEOMETRY_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TASK_SHADER_READ_UNIFORM_BUFFER_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_TASK_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_TASK_SHADER_READ_OTHER_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_MESH_SHADER_READ_UNIFORM_BUFFER_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_MESH_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_MESH_SHADER_READ_OTHER_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_EXT:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_FRAGMENT_DENSITY_MAP_READ_EXT:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_SHADING_RATE_READ_NV:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_FRAGMENT_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_FRAGMENT_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_FRAGMENT_SHADER_READ_COLOR_INPUT_ATTACHMENT:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .access_mask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_FRAGMENT_SHADER_READ_DEPTH_STENCIL_INPUT_ATTACHMENT:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .access_mask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_FRAGMENT_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_COLOR_ATTACHMENT_READ:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access_mask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
        case THSVS_ACCESS_COLOR_ATTACHMENT_ADVANCED_BLENDING_EXT:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_COMPUTE_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_COMPUTE_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_COMPUTE_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_ANY_SHADER_READ_UNIFORM_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_ANY_SHADER_READ_UNIFORM_BUFFER_OR_VERTEX_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .access_mask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_ANY_SHADER_READ_SAMPLED_IMAGE_OR_UNIFORM_TEXEL_BUFFER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        case THSVS_ACCESS_ANY_SHADER_READ_OTHER:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .access_mask = VK_ACCESS_SHADER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TRANSFER_READ:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                .access_mask = VK_ACCESS_TRANSFER_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            };
        case THSVS_ACCESS_HOST_READ:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_HOST_BIT,
                .access_mask = VK_ACCESS_HOST_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_PRESENT:
            return AccessInfo{
                .stage_mask = 0,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            };
        case THSVS_ACCESS_CONDITIONAL_RENDERING_READ_EXT:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ_NV:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_ACCELERATION_STRUCTURE_BUILD_READ_NV:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_END_OF_READ_ACCESS:
            // todo
            return AccessInfo{
                .stage_mask = 0,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_COMMAND_BUFFER_WRITE_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
                .access_mask = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_VERTEX_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TESSELLATION_CONTROL_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TESSELLATION_EVALUATION_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_GEOMETRY_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TASK_SHADER_WRITE_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_MESH_SHADER_WRITE_NV:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TRANSFORM_FEEDBACK_WRITE_EXT:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_FRAGMENT_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_COLOR_ATTACHMENT_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
        case THSVS_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
        case THSVS_ACCESS_DEPTH_ATTACHMENT_WRITE_STENCIL_READ_ONLY:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL};
        case THSVS_ACCESS_STENCIL_ATTACHMENT_WRITE_DEPTH_READ_ONLY:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                .image_layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
            };
        case THSVS_ACCESS_COMPUTE_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_ANY_SHADER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .access_mask = VK_ACCESS_SHADER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_TRANSFER_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                .access_mask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            };
        case THSVS_ACCESS_HOST_PREINITIALIZED:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_HOST_BIT,
                .access_mask = VK_ACCESS_HOST_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_HOST_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_HOST_BIT,
                .access_mask = VK_ACCESS_HOST_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL,
            };
        case THSVS_ACCESS_ACCELERATION_STRUCTURE_BUILD_WRITE_NV:
            // todo
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                .access_mask = 0,
                .image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        case THSVS_ACCESS_COLOR_ATTACHMENT_READ_WRITE:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access_mask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        case THSVS_ACCESS_GENERAL:
            return AccessInfo{
                .stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .access_mask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL};

        default:
            return {};
    }
}

VkImageAspectFlags image_aspect_mask_from_format(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

VkImageAspectFlags image_aspect_mask_from_access_type_and_format(ThsvsAccessType access_type, VkFormat format) {
    auto image_layout = get_access_info(access_type).image_layout;
    switch (image_layout) {
        case VK_IMAGE_LAYOUT_GENERAL:
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return image_aspect_mask_from_format(format);
        default:
            return 0;
    }
}

VkImageUsageFlags image_access_mask_to_usage_flags(VkAccessFlags access_mask) {
    switch (access_mask) {
        case VK_ACCESS_SHADER_READ_BIT:
            return VK_IMAGE_USAGE_SAMPLED_BIT;
        case VK_ACCESS_SHADER_WRITE_BIT:
            return VK_IMAGE_USAGE_STORAGE_BIT;
        case VK_ACCESS_COLOR_ATTACHMENT_READ_BIT:
        case VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT:
            return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        case VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT:
        case VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT:
            return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        case VK_ACCESS_TRANSFER_READ_BIT:
            return VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        case VK_ACCESS_TRANSFER_WRITE_BIT:
            return VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        case VK_ACCESS_MEMORY_READ_BIT:
        case VK_ACCESS_MEMORY_WRITE_BIT:
            return VK_IMAGE_USAGE_STORAGE_BIT;
        default:
            LOGE("Invalid Access Flags {}", access_mask);
    }
    return 0;
}

VkBufferUsageFlags buffer_access_mask_to_usage_flags(VkAccessFlags access_mask) {
    switch (access_mask) {
        case VK_ACCESS_INDIRECT_COMMAND_READ_BIT:
            return VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        case VK_ACCESS_INDEX_READ_BIT:
            return VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        case VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT:
            return VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
        case VK_ACCESS_UNIFORM_READ_BIT:
            return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        case VK_ACCESS_SHADER_READ_BIT:
            return VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
        case VK_ACCESS_SHADER_WRITE_BIT:
            return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        case VK_ACCESS_TRANSFER_READ_BIT:
            return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        case VK_ACCESS_TRANSFER_WRITE_BIT:
            return VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        case VK_ACCESS_MEMORY_READ_BIT:
        case VK_ACCESS_MEMORY_WRITE_BIT:
            return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        default:
            LOGE("Invalid Access Flags {}", access_mask);
    }
    return 0;
}

}// namespace vox::core