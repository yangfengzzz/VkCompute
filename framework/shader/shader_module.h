//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "shader/shader_variant.h"
#include <libshaderc_util/file_finder.h>

namespace vox {
namespace core {
class Device;
}// namespace core

/// Types of shader resources
enum class ShaderResourceType {
    Input,
    InputAttachment,
    Output,
    Image,
    ImageSampler,
    ImageStorage,
    Sampler,
    BufferUniform,
    BufferStorage,
    PushConstant,
    SpecializationConstant,
    All
};

/// This determines the type and method of how descriptor set should be created and bound
enum class ShaderResourceMode {
    Static,
    Dynamic,
    UpdateAfterBind
};

/// A bitmask of qualifiers applied to a resource
struct ShaderResourceQualifiers {
    enum : uint32_t {
        None = 0,
        NonReadable = 1,
        NonWritable = 2,
    };
};

/// Store shader resource data.
/// Used by the shader module.
struct ShaderResource {
    VkShaderStageFlags stages;

    ShaderResourceType type;

    ShaderResourceMode mode;

    uint32_t set;

    uint32_t binding;

    uint32_t location;

    uint32_t input_attachment_index;

    uint32_t vec_size;

    uint32_t columns;

    uint32_t array_size;

    uint32_t offset;

    uint32_t size;

    uint32_t constant_id;

    uint32_t qualifiers;

    std::string name;
};

/**
 * @brief Contains shader code, with an entry point, for a specific shader stage.
 * It is needed by a PipelineLayout to create a Pipeline.
 * ShaderModule can do auto-pairing between shader code and textures.
 * The low level code can change bindings, just keeping the name of the texture.
 * Variants for each texture are also generated, such as HAS_BASE_COLOR_TEX.
 * It works similarly for attribute locations. A current limitation is that only set 0
 * is considered. Uniform buffers are currently hardcoded as well.
 */
class ShaderModule {
public:
    ShaderModule(core::Device &device,
                 VkShaderStageFlagBits stage,
                 const std::string &glsl_source,
                 const std::string &entry_point,
                 const ShaderVariant &shader_variant);

    ShaderModule(core::Device &device,
                 VkShaderStageFlagBits stage,
                 const std::string &spv_source,
                 const std::string &entry_point);

    ShaderModule(const ShaderModule &) = delete;

    ShaderModule(ShaderModule &&other) noexcept;

    ShaderModule &operator=(const ShaderModule &) = delete;

    ShaderModule &operator=(ShaderModule &&) = delete;

    [[nodiscard]] size_t get_id() const;

    [[nodiscard]] VkShaderStageFlagBits get_stage() const;

    [[nodiscard]] const std::string &get_entry_point() const;

    [[nodiscard]] const std::vector<ShaderResource> &get_resources() const;

    [[nodiscard]] const std::string &get_info_log() const;

    [[nodiscard]] const std::vector<uint32_t> &get_binary() const;

    [[nodiscard]] inline const std::string &get_debug_name() const {
        return debug_name;
    }

    inline void set_debug_name(const std::string &name) {
        debug_name = name;
    }

    /**
	 * @brief Flags a resource to use a different method of being bound to the shader
	 * @param resource_name The name of the shader resource
	 * @param resource_mode The mode of how the shader resource will be bound
	 */
    void set_resource_mode(const std::string &resource_name, const ShaderResourceMode &resource_mode);

private:
    core::Device &device;

    /// Shader unique id
    size_t id;

    /// Stage of the shader (vertex, fragment, etc)
    VkShaderStageFlagBits stage{};

    /// Name of the main function
    std::string entry_point;

    /// Human-readable name for the shader
    std::string debug_name;

    /// Compiled source
    std::vector<uint32_t> spirv;

    std::vector<ShaderResource> resources;

    std::string info_log;

    // A FileFinder used to substitute #include directives in the source code.
    shaderc_util::FileFinder include_file_finder_;
};

}// namespace vox
