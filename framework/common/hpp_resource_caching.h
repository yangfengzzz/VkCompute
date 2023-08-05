/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "resource_caching.h"
#include <core/hpp_device.h>
#include <vulkan/vulkan_hash.hpp>

namespace std {
template<typename Key, typename Value>
struct hash<std::map<Key, Value>> {
    size_t operator()(std::map<Key, Value> const &bindings) const {
        size_t result = 0;
        vox::hash_combine(result, bindings.size());
        for (auto &binding : bindings) {
            vox::hash_combine(result, binding.first);
            vox::hash_combine(result, binding.second);
        }
        return result;
    }
};

template<typename T>
struct hash<std::vector<T>> {
    size_t operator()(std::vector<T> const &values) const {
        size_t result = 0;
        vox::hash_combine(result, values.size());
        for (auto const &value : values) {
            vox::hash_combine(result, value);
        }
        return result;
    }
};

template<>
struct hash<vox::common::HPPLoadStoreInfo> {
    size_t operator()(vox::common::HPPLoadStoreInfo const &lsi) const {
        size_t result = 0;
        vox::hash_combine(result, lsi.load_op);
        vox::hash_combine(result, lsi.store_op);
        return result;
    }
};

template<typename T>
struct hash<vox::core::HPPVulkanResource<T>> {
    size_t operator()(const vox::core::HPPVulkanResource<T> &vulkan_resource) const {
        return std::hash<T>()(vulkan_resource.get_handle());
    }
};

template<>
struct hash<vox::core::HPPDescriptorPool> {
    size_t operator()(const vox::core::HPPDescriptorPool &descriptor_pool) const {
        return std::hash<vox::DescriptorPool>()(reinterpret_cast<vox::DescriptorPool const &>(descriptor_pool));
    }
};

template<>
struct hash<vox::core::HPPDescriptorSet> {
    size_t operator()(vox::core::HPPDescriptorSet &descriptor_set) const {
        size_t result = 0;
        vox::hash_combine(result, descriptor_set.get_layout());
        // descriptor_pool ?
        vox::hash_combine(result, descriptor_set.get_buffer_infos());
        vox::hash_combine(result, descriptor_set.get_image_infos());
        vox::hash_combine(result, descriptor_set.get_handle());
        // write_descriptor_sets ?

        return result;
    }
};

template<>
struct hash<vox::core::HPPDescriptorSetLayout> {
    size_t operator()(const vox::core::HPPDescriptorSetLayout &descriptor_set_layout) const {
        return std::hash<vox::DescriptorSetLayout>()(reinterpret_cast<vox::DescriptorSetLayout const &>(descriptor_set_layout));
    }
};

template<>
struct hash<vox::core::HPPImage> {
    size_t operator()(const vox::core::HPPImage &image) const {
        size_t result = 0;
        vox::hash_combine(result, image.get_memory());
        vox::hash_combine(result, image.get_type());
        vox::hash_combine(result, image.get_extent());
        vox::hash_combine(result, image.get_format());
        vox::hash_combine(result, image.get_usage());
        vox::hash_combine(result, image.get_sample_count());
        vox::hash_combine(result, image.get_tiling());
        vox::hash_combine(result, image.get_subresource());
        vox::hash_combine(result, image.get_array_layer_count());
        return result;
    }
};

template<>
struct hash<vox::core::HPPImageView> {
    size_t operator()(const vox::core::HPPImageView &image_view) const {
        size_t result = std::hash<vox::core::HPPVulkanResource<vk::ImageView>>()(image_view);
        vox::hash_combine(result, image_view.get_image());
        vox::hash_combine(result, image_view.get_format());
        vox::hash_combine(result, image_view.get_subresource_range());
        return result;
    }
};

template<>
struct hash<vox::core::HPPRenderPass> {
    size_t operator()(const vox::core::HPPRenderPass &render_pass) const {
        return std::hash<vox::RenderPass>()(reinterpret_cast<vox::RenderPass const &>(render_pass));
    }
};

template<>
struct hash<vox::core::HPPShaderModule> {
    size_t operator()(const vox::core::HPPShaderModule &shader_module) const {
        return std::hash<vox::ShaderModule>()(reinterpret_cast<vox::ShaderModule const &>(shader_module));
    }
};

template<>
struct hash<vox::core::HPPShaderResource> {
    size_t operator()(vox::core::HPPShaderResource const &shader_resource) const {
        size_t result = 0;
        vox::hash_combine(result, shader_resource.stages);
        vox::hash_combine(result, shader_resource.type);
        vox::hash_combine(result, shader_resource.mode);
        vox::hash_combine(result, shader_resource.set);
        vox::hash_combine(result, shader_resource.binding);
        vox::hash_combine(result, shader_resource.location);
        vox::hash_combine(result, shader_resource.input_attachment_index);
        vox::hash_combine(result, shader_resource.vec_size);
        vox::hash_combine(result, shader_resource.columns);
        vox::hash_combine(result, shader_resource.array_size);
        vox::hash_combine(result, shader_resource.offset);
        vox::hash_combine(result, shader_resource.size);
        vox::hash_combine(result, shader_resource.constant_id);
        vox::hash_combine(result, shader_resource.qualifiers);
        vox::hash_combine(result, shader_resource.name);
        return result;
    }
};

template<>
struct hash<vox::core::HPPShaderSource> {
    size_t operator()(const vox::core::HPPShaderSource &shader_source) const {
        return std::hash<vox::ShaderSource>()(reinterpret_cast<vox::ShaderSource const &>(shader_source));
    }
};

template<>
struct hash<vox::core::HPPShaderVariant> {
    size_t operator()(const vox::core::HPPShaderVariant &shader_variant) const {
        return std::hash<vox::ShaderVariant>()(reinterpret_cast<vox::ShaderVariant const &>(shader_variant));
    }
};

template<>
struct hash<vox::core::HPPSubpassInfo> {
    size_t operator()(vox::core::HPPSubpassInfo const &subpass_info) const {
        size_t result = 0;
        vox::hash_combine(result, subpass_info.input_attachments);
        vox::hash_combine(result, subpass_info.output_attachments);
        vox::hash_combine(result, subpass_info.color_resolve_attachments);
        vox::hash_combine(result, subpass_info.disable_depth_stencil_attachment);
        vox::hash_combine(result, subpass_info.depth_stencil_resolve_attachment);
        vox::hash_combine(result, subpass_info.depth_stencil_resolve_mode);
        vox::hash_combine(result, subpass_info.debug_name);
        return result;
    }
};

template<>
struct hash<vox::rendering::HPPAttachment> {
    size_t operator()(const vox::rendering::HPPAttachment &attachment) const {
        size_t result = 0;
        vox::hash_combine(result, attachment.format);
        vox::hash_combine(result, attachment.samples);
        vox::hash_combine(result, attachment.usage);
        vox::hash_combine(result, attachment.initial_layout);
        return result;
    }
};

template<>
struct hash<vox::rendering::HPPPipelineState> {
    size_t operator()(const vox::rendering::HPPPipelineState &pipeline_state) const {
        return std::hash<vox::PipelineState>()(reinterpret_cast<vox::PipelineState const &>(pipeline_state));
    }
};

template<>
struct hash<vox::rendering::HPPRenderTarget> {
    size_t operator()(const vox::rendering::HPPRenderTarget &render_target) const {
        size_t result = 0;
        vox::hash_combine(result, render_target.get_extent());
        for (auto const &view : render_target.get_views()) {
            vox::hash_combine(result, view);
        }
        for (auto const &attachment : render_target.get_attachments()) {
            vox::hash_combine(result, attachment);
        }
        for (auto const &input : render_target.get_input_attachments()) {
            vox::hash_combine(result, input);
        }
        for (auto const &output : render_target.get_output_attachments()) {
            vox::hash_combine(result, output);
        }
        return result;
    }
};

}// namespace std

namespace vox {
/**
 * @brief facade helper functions and structs around the functions and structs in common/resource_caching, providing a vulkan.hpp-based interface
 */

namespace {
template<class T, class... A>
struct HPPRecordHelper {
    size_t record(HPPResourceRecord & /*recorder*/, A &.../*args*/) {
        return 0;
    }

    void index(HPPResourceRecord & /*recorder*/, size_t /*index*/, T & /*resource*/) {}
};

template<class... A>
struct HPPRecordHelper<vox::core::HPPShaderModule, A...> {
    size_t record(HPPResourceRecord &recorder, A &...args) {
        return recorder.register_shader_module(args...);
    }

    void index(HPPResourceRecord &recorder, size_t index, vox::core::HPPShaderModule &shader_module) {
        recorder.set_shader_module(index, shader_module);
    }
};

template<class... A>
struct HPPRecordHelper<vox::core::HPPPipelineLayout, A...> {
    size_t record(HPPResourceRecord &recorder, A &...args) {
        return recorder.register_pipeline_layout(args...);
    }

    void index(HPPResourceRecord &recorder, size_t index, vox::core::HPPPipelineLayout &pipeline_layout) {
        recorder.set_pipeline_layout(index, pipeline_layout);
    }
};

template<class... A>
struct HPPRecordHelper<vox::core::HPPRenderPass, A...> {
    size_t record(HPPResourceRecord &recorder, A &...args) {
        return recorder.register_render_pass(args...);
    }

    void index(HPPResourceRecord &recorder, size_t index, vox::core::HPPRenderPass &render_pass) {
        recorder.set_render_pass(index, render_pass);
    }
};

template<class... A>
struct HPPRecordHelper<vox::core::HPPGraphicsPipeline, A...> {
    size_t record(HPPResourceRecord &recorder, A &...args) {
        return recorder.register_graphics_pipeline(args...);
    }

    void index(HPPResourceRecord &recorder, size_t index, vox::core::HPPGraphicsPipeline &graphics_pipeline) {
        recorder.set_graphics_pipeline(index, graphics_pipeline);
    }
};
}// namespace

template<class T, class... A>
T &request_resource(vox::core::HPPDevice &device, vox::HPPResourceRecord *recorder, std::unordered_map<size_t, T> &resources, A &...args) {
    HPPRecordHelper<T, A...> record_helper;

    size_t hash{0U};
    hash_param(hash, args...);

    auto res_it = resources.find(hash);

    if (res_it != resources.end()) {
        return res_it->second;
    }

    // If we do not have it already, create and cache it
    const char *res_type = typeid(T).name();
    size_t res_id = resources.size();

    LOGD("Building #{} cache object ({})", res_id, res_type);

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

        if (recorder) {
            size_t index = record_helper.record(*recorder, args...);
            record_helper.index(*recorder, index, res_it->second);
        }
#ifndef DEBUG
    } catch (const std::exception &e) {
        LOGE("Creation error for #{} cache object ({})", res_id, res_type);
        throw e;
    }
#endif

    return res_it->second;
}
}// namespace vox
