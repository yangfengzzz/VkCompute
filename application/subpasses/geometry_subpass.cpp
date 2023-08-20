//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "subpasses/geometry_subpass.h"

#include "components/renderer.h"
#include "components/camera.h"
#include "ecs/components_manager.h"
#include "ecs/scene.h"
#include "material/material.h"
#include "mesh/mesh.h"
#include "shader/internal_variant_name.h"

namespace vox {
GeometrySubpass::GeometrySubpass(rendering::RenderContext &render_context, Scene *scene, Camera *camera)
    : Subpass{render_context},
      scene_{scene},
      camera_{camera} {}

void GeometrySubpass::prepare() {}

void GeometrySubpass::draw(core::CommandBuffer &command_buffer) {
    auto compile_variant = ShaderVariant();
    scene_->shader_data.merge_variants(compile_variant, compile_variant);
    if (camera_) {
        camera_->shader_data_.merge_variants(compile_variant, compile_variant);
    }

    std::vector<RenderElement> opaque_queue;
    std::vector<RenderElement> alpha_test_queue;
    std::vector<RenderElement> transparent_queue;
    ComponentsManager::get_singleton().call_render(camera_, opaque_queue, alpha_test_queue, transparent_queue);
    std::sort(opaque_queue.begin(), opaque_queue.end(), compare_from_near_to_far);
    std::sort(alpha_test_queue.begin(), alpha_test_queue.end(), compare_from_near_to_far);
    std::sort(transparent_queue.begin(), transparent_queue.end(), compare_from_far_to_near);

    draw_element(command_buffer, opaque_queue, compile_variant);
    draw_element(command_buffer, alpha_test_queue, compile_variant);
    draw_element(command_buffer, transparent_queue, compile_variant);
}

void GeometrySubpass::draw_element(core::CommandBuffer &command_buffer,
                                   const std::vector<RenderElement> &items,
                                   const ShaderVariant &variant) {
    auto &device = command_buffer.get_device();
    for (auto &element : items) {
        auto macros = variant;
        auto &renderer = element.renderer;
        renderer->update_shader_data();
        renderer->shader_data_.merge_variants(macros, macros);

        auto &material = element.material;
        material->shader_data_.merge_variants(macros, macros);

        auto &sub_mesh = element.sub_mesh;
        auto &mesh = element.mesh;
        core::ScopedDebugLabel submesh_debug_label{command_buffer, mesh->name_.c_str()};

        // pipeline state
        material->rasterization_state_.depth_bias_enable = VK_TRUE;
        command_buffer.set_rasterization_state(material->rasterization_state_);
        command_buffer.set_depth_bias(0.01, 0.01, 1.0);

        auto multisample = material->multisample_state_;
        multisample.rasterization_samples = sample_count;
        command_buffer.set_multisample_state(multisample);
        command_buffer.set_depth_stencil_state(material->depth_stencil_state_);
        command_buffer.set_color_blend_state(material->color_blend_state_);
        command_buffer.set_input_assembly_state(material->input_assembly_state_);

        // shader
        auto &vert_shader_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_VERTEX_BIT,
                                                                                     *material->vertex_source_, macros);
        auto &frag_shader_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                                     *material->fragment_source_, macros);
        std::vector<ShaderModule *> shader_modules{&vert_shader_module, &frag_shader_module};
        auto &pipeline_layout = prepare_pipeline_layout(command_buffer, shader_modules);
        command_buffer.bind_pipeline_layout(pipeline_layout);

        // uniform & texture
        core::DescriptorSetLayout &descriptor_set_layout = pipeline_layout.get_descriptor_set_layout(0);
        scene_->shader_data.bind_data(command_buffer, descriptor_set_layout);
        camera_->shader_data_.bind_data(command_buffer, descriptor_set_layout);
        renderer->shader_data_.bind_data(command_buffer, descriptor_set_layout);
        material->shader_data_.bind_data(command_buffer, descriptor_set_layout);

        // vertex buffer
        command_buffer.set_vertex_input_state(mesh->get_vertex_input_state());
        for (uint32_t j = 0; j < mesh->get_vertex_buffer_count(); j++) {
            const auto kVertexBufferBinding = mesh->get_vertex_buffer(j);
            if (kVertexBufferBinding) {
                std::vector<std::reference_wrapper<const core::Buffer>> buffers;
                buffers.emplace_back(std::ref(*kVertexBufferBinding));
                command_buffer.bind_vertex_buffers(j, buffers, {0});
            }
        }
        // Draw submesh indexed if indices exists
        const auto &index_buffer_binding = mesh->get_index_buffer_binding();
        if (index_buffer_binding) {
            // Bind index buffer of submesh
            command_buffer.bind_index_buffer(index_buffer_binding->get_buffer(), 0, index_buffer_binding->get_index_type());

            // Draw submesh using indexed data
            command_buffer.draw_indexed(sub_mesh->get_count(), mesh->get_instance_count(), sub_mesh->get_start(), 0, 0);
        } else {
            // Draw submesh using vertices only
            command_buffer.draw(sub_mesh->get_count(), mesh->get_instance_count(), 0, 0);
        }
    }
}

bool GeometrySubpass::compare_from_near_to_far(const RenderElement &a, const RenderElement &b) {
    return (a.material->render_queue_ < b.material->render_queue_) ||
           (a.renderer->get_distance_for_sort() < b.renderer->get_distance_for_sort());
}

bool GeometrySubpass::compare_from_far_to_near(const RenderElement &a, const RenderElement &b) {
    return (a.material->render_queue_ < b.material->render_queue_) ||
           (b.renderer->get_distance_for_sort() < a.renderer->get_distance_for_sort());
}

core::PipelineLayout &GeometrySubpass::prepare_pipeline_layout(core::CommandBuffer &command_buffer,
                                                               const std::vector<ShaderModule *> &shader_modules) {
    // Sets any specified resource modes
    for (auto &shader_module : shader_modules) {
        for (auto &resource_mode : resource_mode_map_) {
            shader_module->set_resource_mode(resource_mode.first, resource_mode.second);
        }
    }

    return command_buffer.get_device().get_resource_cache().request_pipeline_layout(shader_modules);
}

}// namespace vox
