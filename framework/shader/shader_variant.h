//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"

namespace vox {
/**
 * @brief Adds support for C style preprocessor macros to glsl shaders
 *        enabling you to define or undefine certain symbols
 */
class ShaderVariant {
public:
    ShaderVariant() = default;

    ShaderVariant(std::string &&preamble, std::vector<std::string> &&processes);

    size_t get_id() const;

    /**
     * Union of two variant collection.
     * @param left - input variant collection
     * @param right - input variant collection
     * @param result - union variant macro collection
     */
    static void union_collection(const ShaderVariant &left, const ShaderVariant &right, ShaderVariant &result);

    /**
	 * @brief Adds a define macro to the shader
	 * @param def String which should go to the right of a define directive
	 */
    void add_define(const std::string &def);

    /**
     * @brief Remove a def macro to the shader
     * @param def String which should go to the right of a define directive
     */
    void remove_define(const std::string &def);

    /**
	 * @brief Specifies the size of a named runtime array for automatic reflection. If already specified, overrides the size.
	 * @param runtime_array_name String under which the runtime array is named in the shader
	 * @param size Integer specifying the wanted size of the runtime array (in number of elements, not size in bytes), used for automatic allocation of buffers.
	 * See get_declared_struct_size_runtime_array() in spirv_cross.h
	 */
    void add_runtime_array_size(const std::string &runtime_array_name, size_t size);

    void set_runtime_array_sizes(const std::unordered_map<std::string, size_t> &sizes);

    std::string get_preamble() const;

    const std::vector<std::string> &get_processes() const;

    const std::unordered_map<std::string, size_t> &get_runtime_array_sizes() const;

    void clear();

private:
    size_t id{};

    std::set<std::string> preambles;

    std::vector<std::string> processes;

    std::unordered_map<std::string, size_t> runtime_array_sizes;

    void update_id();
};

}// namespace vox