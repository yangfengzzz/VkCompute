//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "material/unlit_material.h"

#include "shader/internal_variant_name.h"
#include "shader/shader_manager.h"

namespace vox {
const Color &UnlitMaterial::get_base_color() const { return base_color_; }

void UnlitMaterial::set_base_color(const Color &new_value) {
    base_color_ = new_value;
    shader_data_.set_data(UnlitMaterial::base_color_prop_, base_color_);
}

std::shared_ptr<Texture> UnlitMaterial::get_base_texture() const { return base_texture_; }

void UnlitMaterial::set_base_texture(const std::shared_ptr<Texture> &new_value) {
    if (new_value) {
        BaseMaterial::last_sampler_create_info_.maxLod = static_cast<float>(new_value->get_mipmaps().size());
        set_base_texture(new_value, BaseMaterial::last_sampler_create_info_);
    }
}

void UnlitMaterial::set_base_texture(const std::shared_ptr<Texture> &new_value, const VkSamplerCreateInfo &info) {
    base_texture_ = new_value;
    if (new_value) {
        shader_data_.set_sampled_texture(base_texture_prop_, new_value->get_vk_image_view(),
                                         &device_.get_resource_cache().request_sampler(info));
        shader_data_.add_define(HAS_BASE_TEXTURE);
    } else {
        shader_data_.remove_define(HAS_BASE_TEXTURE);
    }
}

UnlitMaterial::UnlitMaterial(core::Device &device, const std::string &name)
    : BaseMaterial(device, name), base_color_prop_("baseColor"), base_texture_prop_("baseTexture") {
    vertex_source_ = ShaderManager::get_singleton().load_shader("base/unlit.vert", VK_SHADER_STAGE_VERTEX_BIT);
    fragment_source_ = ShaderManager::get_singleton().load_shader("base/unlit.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    shader_data_.add_define(OMIT_NORMAL);

    shader_data_.set_data(base_color_prop_, base_color_);
}

}// namespace vox
