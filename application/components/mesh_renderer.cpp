//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "components/mesh_renderer.h"

#include "ecs/entity.h"
#include "mesh/mesh.h"
#include "mesh/mesh_attributes.h"
#include "shader/internal_variant_name.h"

namespace vox {
MeshRenderer::MeshRenderer(Entity *entity) : Renderer(entity) {}

void MeshRenderer::set_mesh(const MeshPtr &mesh) {
    auto &last_mesh = mesh_;
    if (last_mesh != mesh) {
        if (last_mesh != nullptr) {
            mesh_update_flag_.reset();
        }
        if (mesh != nullptr) {
            mesh_update_flag_ = mesh->register_update_flag();
        }
        mesh_ = mesh;
    }
}

MeshRenderer::MeshPtr MeshRenderer::get_mesh() { return mesh_; }

void MeshRenderer::render(std::vector<RenderElement> &opaque_queue,
                          std::vector<RenderElement> &alpha_test_queue,
                          std::vector<RenderElement> &transparent_queue) {
    if (mesh_ != nullptr) {
        if (mesh_update_flag_->flag_) {
            const auto &vertex_input_state = mesh_->get_vertex_input_state();

            shader_data_.remove_define(HAS_UV);
            shader_data_.remove_define(HAS_NORMAL);
            shader_data_.remove_define(HAS_TANGENT);
            shader_data_.remove_define(HAS_VERTEXCOLOR);

            for (auto attribute : vertex_input_state.attributes) {
                if (attribute.location == (uint32_t)Attributes::UV_0) {
                    shader_data_.add_define(HAS_UV);
                }
                if (attribute.location == (uint32_t)Attributes::NORMAL) {
                    shader_data_.add_define(HAS_NORMAL);
                }
                if (attribute.location == (uint32_t)Attributes::TANGENT) {
                    shader_data_.add_define(HAS_TANGENT);
                }
                if (attribute.location == (uint32_t)Attributes::COLOR_0) {
                    shader_data_.add_define(HAS_VERTEXCOLOR);
                }
            }
            mesh_update_flag_->flag_ = false;
        }

        auto &sub_meshes = mesh_->get_sub_meshes();
        for (size_t i = 0; i < sub_meshes.size(); i++) {
            std::shared_ptr<Material> material;
            if (i < materials_.size()) {
                material = materials_[i];
            } else {
                material = nullptr;
            }
            if (material != nullptr) {
                RenderElement element(this, mesh_, &sub_meshes[i], material);
                push_primitive(element, opaque_queue, alpha_test_queue, transparent_queue);
            }
        }
    }
}

void MeshRenderer::update_bounds(BoundingBox3F &world_bounds) {
    if (mesh_ != nullptr) {
        const auto kLocalBounds = mesh_->bounds_;
        const auto kWorldMatrix = entity_->transform->get_world_matrix();
        world_bounds = kLocalBounds.transform(kWorldMatrix);
    } else {
        world_bounds.lower_corner = Point3F(0, 0, 0);
        world_bounds.upper_corner = Point3F(0, 0, 0);
    }
}

}// namespace vox
