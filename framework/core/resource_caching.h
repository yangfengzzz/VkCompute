//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/descriptor_pool.h"
#include "core/descriptor_set.h"
#include "core/descriptor_set_layout.h"
#include "core/pipeline.h"
#include "core/pipeline_state.h"
#include "rendering/framebuffer.h"
#include "rendering/render_target.h"

#include "common/helpers.h"

inline bool operator==(const VkSamplerCreateInfo &x, const VkSamplerCreateInfo &y) {
    return x.magFilter == y.magFilter && x.minFilter == y.minFilter && x.mipmapMode == y.mipmapMode &&
           x.addressModeU == y.addressModeU && x.addressModeV == y.addressModeV && x.addressModeW == y.addressModeW &&
           x.mipLodBias == y.mipLodBias && x.anisotropyEnable == y.anisotropyEnable &&
           x.maxAnisotropy == y.maxAnisotropy && x.compareEnable == y.compareEnable && x.compareOp == y.compareOp &&
           x.minLod == y.minLod && x.maxLod == y.maxLod && x.borderColor == y.borderColor;
}

namespace std {
template<>
struct hash<VkSamplerCreateInfo> {
    std::size_t operator()(const VkSamplerCreateInfo &sampler) const {
        std::size_t result = 0;

        vox::hash_combine(result, sampler.magFilter);
        vox::hash_combine(result, sampler.minFilter);
        vox::hash_combine(result, sampler.mipmapMode);
        vox::hash_combine(result, sampler.addressModeU);
        vox::hash_combine(result, sampler.addressModeV);
        vox::hash_combine(result, sampler.addressModeW);
        vox::hash_combine(result, sampler.mipLodBias);
        vox::hash_combine(result, sampler.anisotropyEnable);
        vox::hash_combine(result, sampler.maxAnisotropy);
        vox::hash_combine(result, sampler.compareEnable);
        vox::hash_combine(result, sampler.compareOp);
        vox::hash_combine(result, sampler.minLod);
        vox::hash_combine(result, sampler.maxLod);
        vox::hash_combine(result, sampler.borderColor);
        return result;
    }
};

template<>
struct hash<vox::core::DescriptorSetLayout> {
    std::size_t operator()(const vox::core::DescriptorSetLayout &descriptor_set_layout) const {
        std::size_t result = 0;

        vox::hash_combine(result, descriptor_set_layout.get_handle());

        return result;
    }
};

template<>
struct hash<vox::core::DescriptorPool> {
    std::size_t operator()(const vox::core::DescriptorPool &descriptor_pool) const {
        std::size_t result = 0;

        vox::hash_combine(result, descriptor_pool.get_descriptor_set_layout());

        return result;
    }
};

template<>
struct hash<vox::core::PipelineLayout> {
    std::size_t operator()(const vox::core::PipelineLayout &pipeline_layout) const {
        std::size_t result = 0;

        vox::hash_combine(result, pipeline_layout.get_handle());

        return result;
    }
};

template<>
struct hash<vox::core::RenderPass> {
    std::size_t operator()(const vox::core::RenderPass &render_pass) const {
        std::size_t result = 0;

        vox::hash_combine(result, render_pass.get_handle());

        return result;
    }
};

template<>
struct hash<vox::rendering::Attachment> {
    std::size_t operator()(const vox::rendering::Attachment &attachment) const {
        std::size_t result = 0;

        vox::hash_combine(result, static_cast<std::underlying_type<VkFormat>::type>(attachment.format));
        vox::hash_combine(result, static_cast<std::underlying_type<VkSampleCountFlagBits>::type>(attachment.samples));
        vox::hash_combine(result, attachment.usage);
        vox::hash_combine(result, static_cast<std::underlying_type<VkImageLayout>::type>(attachment.initial_layout));

        return result;
    }
};

template<>
struct hash<vox::LoadStoreInfo> {
    std::size_t operator()(const vox::LoadStoreInfo &load_store_info) const {
        std::size_t result = 0;

        vox::hash_combine(result, static_cast<std::underlying_type<VkAttachmentLoadOp>::type>(load_store_info.load_op));
        vox::hash_combine(result, static_cast<std::underlying_type<VkAttachmentStoreOp>::type>(load_store_info.store_op));

        return result;
    }
};

template<>
struct hash<vox::core::SubpassInfo> {
    std::size_t operator()(const vox::core::SubpassInfo &subpass_info) const {
        std::size_t result = 0;

        for (uint32_t output_attachment : subpass_info.output_attachments) {
            vox::hash_combine(result, output_attachment);
        }

        for (uint32_t input_attachment : subpass_info.input_attachments) {
            vox::hash_combine(result, input_attachment);
        }

        for (uint32_t resolve_attachment : subpass_info.color_resolve_attachments) {
            vox::hash_combine(result, resolve_attachment);
        }

        vox::hash_combine(result, subpass_info.disable_depth_stencil_attachment);
        vox::hash_combine(result, subpass_info.depth_stencil_resolve_attachment);
        vox::hash_combine(result, subpass_info.depth_stencil_resolve_mode);

        return result;
    }
};

template<>
struct hash<vox::core::SpecializationConstantState> {
    std::size_t operator()(const vox::core::SpecializationConstantState &specialization_constant_state) const {
        std::size_t result = 0;

        for (const auto &constants : specialization_constant_state.get_specialization_constant_state()) {
            vox::hash_combine(result, constants.first);
            for (const auto data : constants.second) {
                vox::hash_combine(result, data);
            }
        }

        return result;
    }
};

template<>
struct hash<vox::ShaderResource> {
    std::size_t operator()(const vox::ShaderResource &shader_resource) const {
        std::size_t result = 0;

        if (shader_resource.type == vox::ShaderResourceType::Input ||
            shader_resource.type == vox::ShaderResourceType::Output ||
            shader_resource.type == vox::ShaderResourceType::PushConstant ||
            shader_resource.type == vox::ShaderResourceType::SpecializationConstant) {
            return result;
        }

        vox::hash_combine(result, shader_resource.set);
        vox::hash_combine(result, shader_resource.binding);
        vox::hash_combine(result, static_cast<std::underlying_type<vox::ShaderResourceType>::type>(shader_resource.type));
        vox::hash_combine(result, shader_resource.mode);

        return result;
    }
};

template<>
struct hash<VkDescriptorBufferInfo> {
    std::size_t operator()(const VkDescriptorBufferInfo &descriptor_buffer_info) const {
        std::size_t result = 0;

        vox::hash_combine(result, descriptor_buffer_info.buffer);
        vox::hash_combine(result, descriptor_buffer_info.range);
        vox::hash_combine(result, descriptor_buffer_info.offset);

        return result;
    }
};

template<>
struct hash<VkDescriptorImageInfo> {
    std::size_t operator()(const VkDescriptorImageInfo &descriptor_image_info) const {
        std::size_t result = 0;

        vox::hash_combine(result, descriptor_image_info.imageView);
        vox::hash_combine(result, static_cast<std::underlying_type<VkImageLayout>::type>(descriptor_image_info.imageLayout));
        vox::hash_combine(result, descriptor_image_info.sampler);

        return result;
    }
};

template<>
struct hash<VkWriteDescriptorSet> {
    std::size_t operator()(const VkWriteDescriptorSet &write_descriptor_set) const {
        std::size_t result = 0;

        vox::hash_combine(result, write_descriptor_set.dstSet);
        vox::hash_combine(result, write_descriptor_set.dstBinding);
        vox::hash_combine(result, write_descriptor_set.dstArrayElement);
        vox::hash_combine(result, write_descriptor_set.descriptorCount);
        vox::hash_combine(result, write_descriptor_set.descriptorType);

        switch (write_descriptor_set.descriptorType) {
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                for (uint32_t i = 0; i < write_descriptor_set.descriptorCount; i++) {
                    vox::hash_combine(result, write_descriptor_set.pImageInfo[i]);
                }
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                for (uint32_t i = 0; i < write_descriptor_set.descriptorCount; i++) {
                    vox::hash_combine(result, write_descriptor_set.pTexelBufferView[i]);
                }
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                for (uint32_t i = 0; i < write_descriptor_set.descriptorCount; i++) {
                    vox::hash_combine(result, write_descriptor_set.pBufferInfo[i]);
                }
                break;

            default:
                // Not implemented
                break;
        }

        return result;
    }
};

template<>
struct hash<VkVertexInputAttributeDescription> {
    std::size_t operator()(const VkVertexInputAttributeDescription &vertex_attrib) const {
        std::size_t result = 0;

        vox::hash_combine(result, vertex_attrib.binding);
        vox::hash_combine(result, static_cast<std::underlying_type<VkFormat>::type>(vertex_attrib.format));
        vox::hash_combine(result, vertex_attrib.location);
        vox::hash_combine(result, vertex_attrib.offset);

        return result;
    }
};

template<>
struct hash<VkVertexInputBindingDescription> {
    std::size_t operator()(const VkVertexInputBindingDescription &vertex_binding) const {
        std::size_t result = 0;

        vox::hash_combine(result, vertex_binding.binding);
        vox::hash_combine(result, static_cast<std::underlying_type<VkVertexInputRate>::type>(vertex_binding.inputRate));
        vox::hash_combine(result, vertex_binding.stride);

        return result;
    }
};

template<>
struct hash<vox::core::StencilOpState> {
    std::size_t operator()(const vox::core::StencilOpState &stencil) const {
        std::size_t result = 0;

        vox::hash_combine(result, static_cast<std::underlying_type<VkCompareOp>::type>(stencil.compare_op));
        vox::hash_combine(result, static_cast<std::underlying_type<VkStencilOp>::type>(stencil.depth_fail_op));
        vox::hash_combine(result, static_cast<std::underlying_type<VkStencilOp>::type>(stencil.fail_op));
        vox::hash_combine(result, static_cast<std::underlying_type<VkStencilOp>::type>(stencil.pass_op));

        return result;
    }
};

template<>
struct hash<VkExtent2D> {
    size_t operator()(const VkExtent2D &extent) const {
        size_t result = 0;

        vox::hash_combine(result, extent.width);
        vox::hash_combine(result, extent.height);

        return result;
    }
};

template<>
struct hash<VkOffset2D> {
    size_t operator()(const VkOffset2D &offset) const {
        size_t result = 0;

        vox::hash_combine(result, offset.x);
        vox::hash_combine(result, offset.y);

        return result;
    }
};

template<>
struct hash<VkRect2D> {
    size_t operator()(const VkRect2D &rect) const {
        size_t result = 0;

        vox::hash_combine(result, rect.extent);
        vox::hash_combine(result, rect.offset);

        return result;
    }
};

template<>
struct hash<VkViewport> {
    size_t operator()(const VkViewport &viewport) const {
        size_t result = 0;

        vox::hash_combine(result, viewport.width);
        vox::hash_combine(result, viewport.height);
        vox::hash_combine(result, viewport.maxDepth);
        vox::hash_combine(result, viewport.minDepth);
        vox::hash_combine(result, viewport.x);
        vox::hash_combine(result, viewport.y);

        return result;
    }
};

template<>
struct hash<vox::core::ColorBlendAttachmentState> {
    std::size_t operator()(const vox::core::ColorBlendAttachmentState &color_blend_attachment) const {
        std::size_t result = 0;

        vox::hash_combine(result, static_cast<std::underlying_type<VkBlendOp>::type>(color_blend_attachment.alpha_blend_op));
        vox::hash_combine(result, color_blend_attachment.blend_enable);
        vox::hash_combine(result, static_cast<std::underlying_type<VkBlendOp>::type>(color_blend_attachment.color_blend_op));
        vox::hash_combine(result, color_blend_attachment.color_write_mask);
        vox::hash_combine(result, static_cast<std::underlying_type<VkBlendFactor>::type>(color_blend_attachment.dst_alpha_blend_factor));
        vox::hash_combine(result, static_cast<std::underlying_type<VkBlendFactor>::type>(color_blend_attachment.dst_color_blend_factor));
        vox::hash_combine(result, static_cast<std::underlying_type<VkBlendFactor>::type>(color_blend_attachment.src_alpha_blend_factor));
        vox::hash_combine(result, static_cast<std::underlying_type<VkBlendFactor>::type>(color_blend_attachment.src_color_blend_factor));

        return result;
    }
};

template<>
struct hash<vox::rendering::RenderTarget> {
    std::size_t operator()(const vox::rendering::RenderTarget &render_target) const {
        std::size_t result = 0;

        for (auto &view : render_target.get_views()) {
            vox::hash_combine(result, view.get_handle());
            vox::hash_combine(result, view.get_image().get_handle());
        }

        return result;
    }
};

template<>
struct hash<vox::core::PipelineState> {
    std::size_t operator()(const vox::core::PipelineState &pipeline_state) const {
        std::size_t result = 0;

        vox::hash_combine(result, pipeline_state.get_pipeline_layout().get_handle());

        // For graphics only
        if (auto render_pass = pipeline_state.get_render_pass()) {
            vox::hash_combine(result, render_pass->get_handle());
        }

        vox::hash_combine(result, pipeline_state.get_specialization_constant_state());

        vox::hash_combine(result, pipeline_state.get_subpass_index());

        for (auto shader_module : pipeline_state.get_pipeline_layout().get_shader_modules()) {
            vox::hash_combine(result, shader_module->get_id());
        }

        // VkPipelineVertexInputStateCreateInfo
        for (auto &attribute : pipeline_state.get_vertex_input_state().attributes) {
            vox::hash_combine(result, attribute);
        }

        for (auto &binding : pipeline_state.get_vertex_input_state().bindings) {
            vox::hash_combine(result, binding);
        }

        // VkPipelineInputAssemblyStateCreateInfo
        vox::hash_combine(result, pipeline_state.get_input_assembly_state().primitive_restart_enable);
        vox::hash_combine(result, static_cast<std::underlying_type<VkPrimitiveTopology>::type>(pipeline_state.get_input_assembly_state().topology));

        //VkPipelineViewportStateCreateInfo
        vox::hash_combine(result, pipeline_state.get_viewport_state().viewport_count);
        vox::hash_combine(result, pipeline_state.get_viewport_state().scissor_count);

        // VkPipelineRasterizationStateCreateInfo
        vox::hash_combine(result, pipeline_state.get_rasterization_state().cull_mode);
        vox::hash_combine(result, pipeline_state.get_rasterization_state().depth_bias_enable);
        vox::hash_combine(result, pipeline_state.get_rasterization_state().depth_clamp_enable);
        vox::hash_combine(result, static_cast<std::underlying_type<VkFrontFace>::type>(pipeline_state.get_rasterization_state().front_face));
        vox::hash_combine(result, static_cast<std::underlying_type<VkPolygonMode>::type>(pipeline_state.get_rasterization_state().polygon_mode));
        vox::hash_combine(result, pipeline_state.get_rasterization_state().rasterizer_discard_enable);

        // VkPipelineMultisampleStateCreateInfo
        vox::hash_combine(result, pipeline_state.get_multisample_state().alpha_to_coverage_enable);
        vox::hash_combine(result, pipeline_state.get_multisample_state().alpha_to_one_enable);
        vox::hash_combine(result, pipeline_state.get_multisample_state().min_sample_shading);
        vox::hash_combine(result, static_cast<std::underlying_type<VkSampleCountFlagBits>::type>(pipeline_state.get_multisample_state().rasterization_samples));
        vox::hash_combine(result, pipeline_state.get_multisample_state().sample_shading_enable);
        vox::hash_combine(result, pipeline_state.get_multisample_state().sample_mask);

        // VkPipelineDepthStencilStateCreateInfo
        vox::hash_combine(result, pipeline_state.get_depth_stencil_state().back);
        vox::hash_combine(result, pipeline_state.get_depth_stencil_state().depth_bounds_test_enable);
        vox::hash_combine(result, static_cast<std::underlying_type<VkCompareOp>::type>(pipeline_state.get_depth_stencil_state().depth_compare_op));
        vox::hash_combine(result, pipeline_state.get_depth_stencil_state().depth_test_enable);
        vox::hash_combine(result, pipeline_state.get_depth_stencil_state().depth_write_enable);
        vox::hash_combine(result, pipeline_state.get_depth_stencil_state().front);
        vox::hash_combine(result, pipeline_state.get_depth_stencil_state().stencil_test_enable);

        // VkPipelineColorBlendStateCreateInfo
        vox::hash_combine(result, static_cast<std::underlying_type<VkLogicOp>::type>(pipeline_state.get_color_blend_state().logic_op));
        vox::hash_combine(result, pipeline_state.get_color_blend_state().logic_op_enable);

        for (auto &attachment : pipeline_state.get_color_blend_state().attachments) {
            vox::hash_combine(result, attachment);
        }

        return result;
    }
};
}// namespace std

namespace vox::core {
namespace {
template<typename T>
inline void hash_param(size_t &seed, const T &value) {
    hash_combine(seed, value);
}

template<>
inline void hash_param(size_t & /*seed*/, const VkPipelineCache & /*value*/) {
}

template<>
inline void hash_param<std::vector<uint8_t>>(
    size_t &seed,
    const std::vector<uint8_t> &value) {
    hash_combine(seed, std::string{value.begin(), value.end()});
}

template<>
inline void hash_param<std::vector<rendering::Attachment>>(
    size_t &seed,
    const std::vector<rendering::Attachment> &value) {
    for (auto &attachment : value) {
        hash_combine(seed, attachment);
    }
}

template<>
inline void hash_param<std::vector<LoadStoreInfo>>(
    size_t &seed,
    const std::vector<LoadStoreInfo> &value) {
    for (auto &load_store_info : value) {
        hash_combine(seed, load_store_info);
    }
}

template<>
inline void hash_param<std::vector<SubpassInfo>>(
    size_t &seed,
    const std::vector<SubpassInfo> &value) {
    for (auto &subpass_info : value) {
        hash_combine(seed, subpass_info);
    }
}

template<>
inline void hash_param<std::vector<ShaderModule *>>(
    size_t &seed,
    const std::vector<ShaderModule *> &value) {
    for (auto &shader_module : value) {
        hash_combine(seed, shader_module->get_id());
    }
}

template<>
inline void hash_param<std::vector<ShaderResource>>(
    size_t &seed,
    const std::vector<ShaderResource> &value) {
    for (auto &resource : value) {
        hash_combine(seed, resource);
    }
}

template<>
inline void hash_param<std::map<uint32_t, std::map<uint32_t, VkDescriptorBufferInfo>>>(
    size_t &seed,
    const std::map<uint32_t, std::map<uint32_t, VkDescriptorBufferInfo>> &value) {
    for (auto &binding_set : value) {
        hash_combine(seed, binding_set.first);

        for (auto &binding_element : binding_set.second) {
            hash_combine(seed, binding_element.first);
            hash_combine(seed, binding_element.second);
        }
    }
}

template<>
inline void hash_param<std::map<uint32_t, std::map<uint32_t, VkDescriptorImageInfo>>>(
    size_t &seed,
    const std::map<uint32_t, std::map<uint32_t, VkDescriptorImageInfo>> &value) {
    for (auto &binding_set : value) {
        hash_combine(seed, binding_set.first);

        for (auto &binding_element : binding_set.second) {
            hash_combine(seed, binding_element.first);
            hash_combine(seed, binding_element.second);
        }
    }
}

template<typename T, typename... Args>
inline void hash_param(size_t &seed, const T &first_arg, const Args &...args) {
    hash_param(seed, first_arg);

    hash_param(seed, args...);
}

}// namespace

template<class T, class... A>
T &request_resource(Device &device, std::unordered_map<std::size_t, T> &resources, A &...args) {
    std::size_t hash{0U};
    hash_param(hash, args...);

    auto res_it = resources.find(hash);

    if (res_it != resources.end()) {
        return res_it->second;
    }

    // If we do not have it already, create and cache it
    const char *res_type = typeid(T).name();
    size_t res_id = resources.size();

    LOGD("Building #{} cache object ({})", res_id, res_type)

// Only error handle in release
#ifndef DEBUG
    try {
#endif
        T resource(device, args...);

        auto res_ins_it = resources.emplace(hash, std::move(resource));

        if (!res_ins_it.second) {
            throw std::runtime_error{std::string{"Insertion error for #"} + std::to_string(res_id) + "cache object (" + res_type + ")"};
        }

        res_it = res_ins_it.first;

#ifndef DEBUG
    } catch (const std::exception &e) {
        LOGE("Creation error for #{} cache object ({})", res_id, res_type);
        throw e;
    }
#endif

    return res_it->second;
}

}// namespace vox::core
// namespace vox::core
