//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "shader/shader_module.h"

#include "common/logging.h"
#include "common/filesystem.h"
#include <shaderc/shaderc.hpp>
#include "shader/file_includer.h"
#include "shader/spirv_reflection.h"
#include "core/device.h"

namespace vox {

inline std::vector<uint8_t> convert_to_bytes(std::vector<std::string> &lines) {
    std::vector<uint8_t> bytes;

    for (auto &line : lines) {
        line += "\n";
        std::vector<uint8_t> line_bytes(line.begin(), line.end());
        bytes.insert(bytes.end(), line_bytes.begin(), line_bytes.end());
    }

    return bytes;
}

ShaderModule::ShaderModule(core::Device &device, VkShaderStageFlagBits stage, const std::string &glsl_source,
                           const std::string &entry_point, const ShaderVariant &shader_variant)
    : device{device},
      stage{stage},
      entry_point{entry_point} {
    debug_name = fmt::format("{} [variant {:X}] [entrypoint {}]",
                             glsl_source, shader_variant.get_id(), entry_point);

    // Compiling from GLSL source requires the entry point
    if (entry_point.empty()) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    auto source = fs::read_shader(glsl_source);

    // Check if application is passing in GLSL source code to compile to SPIR-V
    if (source.empty()) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // todo variant
    options.AddMacroDefinition("TYPE", "vec4");

    include_file_finder_.search_path().emplace_back(fs::path::get(fs::path::Type::Shaders));
    auto includer = std::make_unique<FileIncluder>(&include_file_finder_);
    options.SetIncluder(std::move(includer));
#ifdef VKB_VULKAN_DEBUG
    options.SetOptimizationLevel(shaderc_optimization_level_zero);
#else
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif

    shaderc_shader_kind kind{};
    switch (stage) {
        case VK_SHADER_STAGE_COMPUTE_BIT:
            kind = shaderc_glsl_compute_shader;
            break;
        case VK_SHADER_STAGE_VERTEX_BIT:
            kind = shaderc_glsl_vertex_shader;
            break;
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
            kind = shaderc_glsl_tess_control_shader;
            break;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
            kind = shaderc_glsl_tess_evaluation_shader;
            break;
        case VK_SHADER_STAGE_GEOMETRY_BIT:
            kind = shaderc_glsl_geometry_shader;
            break;
        case VK_SHADER_STAGE_FRAGMENT_BIT:
            kind = shaderc_glsl_fragment_shader;
            break;
        case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
            kind = shaderc_glsl_raygen_shader;
            break;
        case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
            kind = shaderc_glsl_anyhit_shader;
            break;
        case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
            kind = shaderc_glsl_closesthit_shader;
            break;
        case VK_SHADER_STAGE_MISS_BIT_KHR:
            kind = shaderc_glsl_miss_shader;
            break;
        case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
            kind = shaderc_glsl_intersection_shader;
            break;
        case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
            kind = shaderc_glsl_callable_shader;
            break;
        case VK_SHADER_STAGE_TASK_BIT_EXT:
            kind = shaderc_glsl_task_shader;
            break;
        case VK_SHADER_STAGE_MESH_BIT_EXT:
            kind = shaderc_glsl_mesh_shader;
            break;
        case VK_SHADER_STAGE_SUBPASS_SHADING_BIT_HUAWEI:
        case VK_SHADER_STAGE_CLUSTER_CULLING_BIT_HUAWEI:
        case VK_SHADER_STAGE_ALL_GRAPHICS:
        case VK_SHADER_STAGE_ALL:
            kind = shaderc_glsl_infer_from_source;
            break;
    }

    // compile
    options.SetGenerateDebugInfo();// keep reflection data
    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source.c_str(), source.size(), kind,
                                  glsl_source.c_str(), options);
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOGD(module.GetErrorMessage());
    }

    spirv = {module.cbegin(), module.cend()};
    // Reflect all shader resources
    if (!SPIRVReflection::reflect_shader_resources(stage, spirv, resources, shader_variant)) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    // Generate a unique id, determined by source and variant
    std::hash<std::string> hasher{};
    id = hasher(std::string{reinterpret_cast<const char *>(spirv.data()),
                            reinterpret_cast<const char *>(spirv.data() + spirv.size())});
}

ShaderModule::ShaderModule(core::Device &device,
                           VkShaderStageFlagBits stage,
                           const std::string &spv_source,
                           const std::string &entry_point)
    : device{device},
      stage{stage},
      entry_point{entry_point} {
    debug_name = fmt::format("{} [entrypoint {}]", spv_source, entry_point);

    // Compiling from GLSL source requires the entry point
    if (entry_point.empty()) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    spirv = fs::read_spv(spv_source);
    SPIRVReflection spirv_reflection;

    // Reflect all shader resources
    if (!spirv_reflection.reflect_shader_resources(stage, spirv, resources, {})) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    // Generate a unique id, determined by source and variant
    std::hash<std::string> hasher{};
    id = hasher(std::string{reinterpret_cast<const char *>(spirv.data()),
                            reinterpret_cast<const char *>(spirv.data() + spirv.size())});
}

ShaderModule::ShaderModule(ShaderModule &&other) noexcept
    : device{other.device},
      id{other.id},
      stage{other.stage},
      entry_point{other.entry_point},
      debug_name{other.debug_name},
      spirv{other.spirv},
      resources{other.resources},
      info_log{other.info_log} {
    other.stage = {};
}

size_t ShaderModule::get_id() const {
    return id;
}

VkShaderStageFlagBits ShaderModule::get_stage() const {
    return stage;
}

const std::string &ShaderModule::get_entry_point() const {
    return entry_point;
}

const std::vector<ShaderResource> &ShaderModule::get_resources() const {
    return resources;
}

const std::string &ShaderModule::get_info_log() const {
    return info_log;
}

const std::vector<uint32_t> &ShaderModule::get_binary() const {
    return spirv;
}

void ShaderModule::set_resource_mode(const std::string &resource_name, const ShaderResourceMode &resource_mode) {
    auto it = std::find_if(resources.begin(), resources.end(),
                           [&resource_name](const ShaderResource &resource) { return resource.name == resource_name; });

    if (it != resources.end()) {
        if (resource_mode == ShaderResourceMode::Dynamic) {
            if (it->type == ShaderResourceType::BufferUniform || it->type == ShaderResourceType::BufferStorage) {
                it->mode = resource_mode;
            } else {
                LOGW("Resource `{}` does not support dynamic.", resource_name)
            }
        } else {
            it->mode = resource_mode;
        }
    } else {
        LOGW("Resource `{}` not found for shader.", resource_name)
    }
}

}// namespace vox
