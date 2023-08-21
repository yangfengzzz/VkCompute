//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "shader/shader_module.h"

#include "common/logging.h"
#include "common/filesystem.h"
#include "shader/glsl_compiler.h"
#include "shader/spirv_reflection.h"
#include "core/device.h"

namespace vox {
/**
 * @brief Pre-compiles project shader files to include header code
 * @param source The shader file
 * @returns A byte array of the final shader
 */
inline std::vector<std::string> precompile_shader(const std::string &source) {
    std::vector<std::string> final_file;

    auto lines = split(source, '\n');

    for (auto &line : lines) {
        if (line.find("#include \"") == 0) {
            // Include paths are relative to the base shader directory
            std::string include_path = line.substr(10);
            size_t last_quote = include_path.find("\"");
            if (!include_path.empty() && last_quote != std::string::npos) {
                include_path = include_path.substr(0, last_quote);
            }

            auto include_file = precompile_shader(fs::read_shader(include_path));
            for (auto &include_file_line : include_file) {
                final_file.push_back(include_file_line);
            }
        } else {
            final_file.push_back(line);
        }
    }

    return final_file;
}

inline std::vector<uint8_t> convert_to_bytes(std::vector<std::string> &lines) {
    std::vector<uint8_t> bytes;

    for (auto &line : lines) {
        line += "\n";
        std::vector<uint8_t> line_bytes(line.begin(), line.end());
        bytes.insert(bytes.end(), line_bytes.begin(), line_bytes.end());
    }

    return bytes;
}

ShaderModule::ShaderModule(core::Device &device, VkShaderStageFlagBits stage, const ShaderSource &glsl_source,
                           const std::string &entry_point, const ShaderVariant &shader_variant)
    : device{device},
      stage{stage},
      entry_point{entry_point} {
    debug_name = fmt::format("{} [variant {:X}] [entrypoint {}]",
                             glsl_source.get_filename(), shader_variant.get_id(), entry_point);

    // Compiling from GLSL source requires the entry point
    if (entry_point.empty()) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    auto &source = glsl_source.get_source();

    // Check if application is passing in GLSL source code to compile to SPIR-V
    if (source.empty()) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    // Precompile source into the final spirv bytecode
    auto glsl_final_source = precompile_shader(source);

    // Compile the GLSL source
    GLSLCompiler glsl_compiler;

    if (!glsl_compiler.compile_to_spirv(stage, convert_to_bytes(glsl_final_source), entry_point, shader_variant, spirv, info_log)) {
        LOGE("Shader compilation failed for shader \"{}\"", glsl_source.get_filename())
        LOGE("{}", info_log)
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    SPIRVReflection spirv_reflection;

    // Reflect all shader resources
    if (!spirv_reflection.reflect_shader_resources(stage, spirv, resources, shader_variant)) {
        throw VulkanException{VK_ERROR_INITIALIZATION_FAILED};
    }

    // Generate a unique id, determined by source and variant
    std::hash<std::string> hasher{};
    id = hasher(std::string{reinterpret_cast<const char *>(spirv.data()),
                            reinterpret_cast<const char *>(spirv.data() + spirv.size())});
}

ShaderModule::ShaderModule(ShaderModule &&other)
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
    auto it = std::find_if(resources.begin(), resources.end(), [&resource_name](const ShaderResource &resource) { return resource.name == resource_name; });

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

ShaderSource::ShaderSource(const std::string &filename) : filename{filename},
                                                          source{fs::read_shader(filename)} {
    std::hash<std::string> hasher{};
    id = hasher(std::string{this->source.cbegin(), this->source.cend()});
}

size_t ShaderSource::get_id() const {
    return id;
}

const std::string &ShaderSource::get_filename() const {
    return filename;
}

void ShaderSource::set_source(const std::string &source_) {
    source = source_;
    std::hash<std::string> hasher{};
    id = hasher(std::string{this->source.cbegin(), this->source.cend()});
}

const std::string &ShaderSource::get_source() const {
    return source;
}

}// namespace vox
