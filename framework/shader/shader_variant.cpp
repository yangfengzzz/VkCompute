//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "shader/shader_variant.h"

namespace vox {
ShaderVariant::ShaderVariant(std::string &&preamble, std::vector<std::string> &&processes)
    : processes{std::move(processes)} {
    auto splits = split(preamble, "\n");
    for (const std::string &split : splits) {
        preambles.insert(split);
    }

    update_id();
}

size_t ShaderVariant::get_id() const {
    return id;
}

void ShaderVariant::union_collection(const ShaderVariant &left, const ShaderVariant &right, ShaderVariant &result) {
    std::set<std::string> collect;
    for (const std::string &def : left.get_processes()) {
        collect.insert(def);
    }

    for (const std::string &def : right.get_processes()) {
        collect.insert(def);
    }

    for (const std::string &def : collect) {
        std::string tmp_def(def, 1);
        result.add_define(tmp_def);
    }
}

void ShaderVariant::add_define(const std::string &def) {
    auto iter = std::find(processes.begin(), processes.end(), "D" + def);
    if (iter == processes.end()) {
        processes.push_back("D" + def);
    }

    std::string tmp_def = def;

    // The "=" needs to turn into a space
    size_t pos_equal = tmp_def.find_first_of('=');
    if (pos_equal != std::string::npos) {
        tmp_def[pos_equal] = ' ';
    }

    preambles.insert("#define " + tmp_def + "\n");

    update_id();
}

void ShaderVariant::remove_define(const std::string &def) {
    std::string process = "D" + def;
    processes.erase(std::remove(processes.begin(), processes.end(), process), processes.end());

    std::string tmp_def = def;
    // The "=" needs to turn into a space
    size_t pos_equal = tmp_def.find_first_of('=');
    if (pos_equal != std::string::npos) {
        tmp_def[pos_equal] = ' ';
    }
    tmp_def = "#define " + tmp_def + "\n";
    auto iter = preambles.find(tmp_def);
    if (iter != preambles.end()) {
        preambles.erase(iter);
    }

    update_id();
}

void ShaderVariant::add_runtime_array_size(const std::string &runtime_array_name, size_t size) {
    if (runtime_array_sizes.find(runtime_array_name) == runtime_array_sizes.end()) {
        runtime_array_sizes.insert({runtime_array_name, size});
    } else {
        runtime_array_sizes[runtime_array_name] = size;
    }
}

void ShaderVariant::set_runtime_array_sizes(const std::unordered_map<std::string, size_t> &sizes) {
    this->runtime_array_sizes = sizes;
}

std::string ShaderVariant::get_preamble() const {
    std::string preamble_;
    std::for_each(preambles.begin(), preambles.end(), [&](const std::string &p) { preamble_ += p; });
    return preamble_;
}

const std::vector<std::string> &ShaderVariant::get_processes() const {
    return processes;
}

const std::unordered_map<std::string, size_t> &ShaderVariant::get_runtime_array_sizes() const {
    return runtime_array_sizes;
}

void ShaderVariant::clear() {
    preambles.clear();
    processes.clear();
    runtime_array_sizes.clear();
    update_id();
}

void ShaderVariant::update_id() {
    id = 0;
    std::for_each(preambles.begin(), preambles.end(),
                  [&](const std::string &preamble) { hash_combine(id, preamble); });
}

}// namespace vox