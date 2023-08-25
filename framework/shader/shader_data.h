//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/buffer_pool.h"
#include "core/buffer.h"
#include "core/command_buffer.h"
#include "core/sampled_image.h"
#include "shader/shader_module.h"

namespace vox {
/**
 * Shader data collection,Correspondence includes shader properties data and macros data.
 */
class ShaderData {
public:
    explicit ShaderData(core::Device &device);

    void bind_data(core::CommandBuffer &command_buffer, core::DescriptorSetLayout &descriptor_set_layout);

    void bind_specialization_constant(core::CommandBuffer &command_buffer, ShaderModule &shader);

    void set_buffer_functor(const std::string &property_name, const std::function<core::Buffer *()> &functor);

    void set_data(const std::string &property_name, core::BufferAllocation &&value);

    template<typename T>
    void set_data(const std::string &property_name, T &value) {
        auto iter = shader_buffers_.find(property_name);
        if (iter == shader_buffers_.end()) {
            shader_buffers_.insert(std::make_pair(
                property_name,
                std::make_unique<core::Buffer>(device_, sizeof(T), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                               VMA_MEMORY_USAGE_CPU_TO_GPU)));
        }
        iter = shader_buffers_.find(property_name);
        iter->second->update(&value, sizeof(T));
    }

    template<typename T>
    void set_data(const std::string &property_name, std::vector<T> &value) {
        auto iter = shader_buffers_.find(property_name);
        if (iter == shader_buffers_.end()) {
            shader_buffers_.insert(std::make_pair(
                property_name,
                std::make_unique<core::Buffer>(device_, sizeof(T) * value.size(),
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU)));
        }
        iter = shader_buffers_.find(property_name);
        iter->second->update(value.data(), sizeof(T) * value.size());
    }

    template<typename T, size_t N>
    void set_data(const std::string &property_name, std::array<T, N> &value) {
        auto iter = shader_buffers_.find(property_name);
        if (iter == shader_buffers_.end()) {
            shader_buffers_.insert(std::make_pair(
                property_name,
                std::make_unique<core::Buffer>(device_, sizeof(T) * N, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                               VMA_MEMORY_USAGE_CPU_TO_GPU)));
        }
        iter = shader_buffers_.find(property_name);
        iter->second->update(value.data(), sizeof(T) * N);
    }

public:
    void set_sampled_texture(const std::string &texture_name, const core::ImageView &image_view, core::Sampler *sampler);

    void set_storage_texture(const std::string &texture_name, const core::ImageView &image_view);

    [[nodiscard]] inline const std::unordered_map<std::string, std::unique_ptr<core::SampledImage>> &sampled_textures() const {
        return sampled_textures_;
    }

    [[nodiscard]] inline const std::unordered_map<std::string, std::unique_ptr<core::SampledImage>> &storage_textures() const {
        return storage_textures_;
    }

public:
    /**
     * @brief Adds a define macro to the shader
     * @param def String which should go to the right of a define directive
     */
    void add_define(const std::string &def);

    /**
     * @brief Adds an undef macro to the shader
     * @param undef String which should go to the right of an undef directive
     */
    void remove_define(const std::string &undef);

private:
    core::Device &device_;
    std::unordered_map<std::string, core::BufferAllocation> shader_buffer_pools_{};
    std::unordered_map<std::string, std::function<core::Buffer *()>> shader_buffer_functors_{};
    std::unordered_map<std::string, std::unique_ptr<core::Buffer>> shader_buffers_{};
    std::unordered_map<std::string, std::unique_ptr<core::SampledImage>> sampled_textures_{};
    std::unordered_map<std::string, std::unique_ptr<core::SampledImage>> storage_textures_{};

    // Map tracking state of the Specialization Constants
    std::map<std::string, std::vector<uint8_t>> specialization_constant_state;
};

}// namespace vox
