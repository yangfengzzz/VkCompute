//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "base/singleton.h"
#include "framework/shader/shader_module.h"

namespace vox {
class ShaderManager : public Singleton<ShaderManager> {
public:
    static ShaderManager &get_singleton();

    static ShaderManager *get_singleton_ptr();

    explicit ShaderManager(core::Device &device);

    /**
     * @brief Loads shader source
     */
    std::shared_ptr<ShaderModule> load_shader(const std::string &file, VkShaderStageFlagBits stage,
                                              const std::string &entry_point = "main",
                                              const ShaderVariant &shader_variant = {});

    void collect_garbage();

private:
    core::Device &device_;

    std::unordered_map<std::string, std::shared_ptr<ShaderModule>> shader_pool_;
};

template<>
inline ShaderManager *Singleton<ShaderManager>::ms_singleton{nullptr};

}// namespace vox
