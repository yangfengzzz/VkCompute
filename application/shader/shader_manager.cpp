//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "shader/shader_manager.h"

namespace vox {
ShaderManager *ShaderManager::get_singleton_ptr() { return ms_singleton; }

ShaderManager &ShaderManager::get_singleton() {
    assert(ms_singleton);
    return (*ms_singleton);
}

ShaderManager::ShaderManager(core::Device &device):
device_{device} {

}

std::shared_ptr<ShaderModule> ShaderManager::load_shader(const std::string &file, VkShaderStageFlagBits stage,
                                                         const std::string &entry_point,
                                                         const ShaderVariant &shader_variant) {
    auto iter = shader_pool_.find(file);
    if (iter != shader_pool_.end() && iter->second != nullptr) {
        return iter->second;
    } else {
        auto shader = std::make_shared<ShaderModule>(device_, stage, file, entry_point, shader_variant);
        shader_pool_.insert(std::make_pair(file, shader));
        return shader;
    }
}

void ShaderManager::collect_garbage() {
    for (auto &shader : shader_pool_) {
        if (shader.second.use_count() == 1) {
            shader.second.reset();
        }
    }
}

}// namespace vox
