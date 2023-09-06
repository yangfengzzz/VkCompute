//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "apps/atomic_compute_app.h"

#include "application/components/camera.h"
#include "application/controls/orbit_control.h"
#include "application/material/base_material.h"
#include "application/components/mesh_renderer.h"
#include "application/mesh/primitive_mesh.h"
#include "application/ecs/entity.h"

namespace vox {
class AtomicMaterial : public BaseMaterial {
public:
    explicit AtomicMaterial(core::Device &device) : BaseMaterial(device, "atomicRender") {
        vertex_source_ = ShaderManager::get_singleton().load_shader("base/unlit.vert", VK_SHADER_STAGE_VERTEX_BIT);
        fragment_source_ = ShaderManager::get_singleton().load_shader("base/compute/atomic_counter.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    }
};

// MARK: - AtomicComputeApp
Camera *AtomicComputeApp::load_scene() {
    auto scene = SceneManager::get_singleton().get_current_scene();
    auto root_entity = scene->create_root_entity();

    auto camera_entity = root_entity->create_child();
    camera_entity->transform->set_position(10, 10, 10);
    camera_entity->transform->look_at(Point3F(0, 0, 0));
    auto main_camera = camera_entity->add_component<Camera>();
    camera_entity->add_component<control::OrbitControl>();

    // init point light
    auto light = root_entity->create_child("light");
    light->transform->set_position(0, 3, 0);
    auto point_light = light->add_component<PointLight>();
    point_light->intensity_ = 0.3;

    auto cube_entity = root_entity->create_child();
    auto renderer = cube_entity->add_component<MeshRenderer>();
    renderer->set_mesh(PrimitiveMesh::create_cuboid(1));
    material_ = std::make_shared<AtomicMaterial>(*device);
    renderer->set_material(material_);

    {
        auto buffer_desc = core::BufferDesc{.size = sizeof(uint32_t),
                                            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            .memory_usage = VMA_MEMORY_USAGE_GPU_ONLY};
        atomic_buffer_ = std::make_unique<core::Buffer>(*device, buffer_desc);
        material_->shader_data_.set_buffer_functor("atomicCounter", [this]() -> core::Buffer * { return atomic_buffer_.get(); });

        auto retained_resource = framegraph.add_retained_resource(
            "Backbuffer",
            buffer_desc,
            THSVS_ACCESS_COMPUTE_SHADER_WRITE,
            atomic_buffer_.get());

        struct ComputeTaskData {
            fg::BufferResource *output;
        };
        framegraph.add_render_task<ComputeTaskData>(
            "atomic compute",
            [&](ComputeTaskData &data, fg::RenderTaskBuilder &builder) { data.output = builder.write(retained_resource, THSVS_ACCESS_COMPUTE_SHADER_WRITE); },
            [&](const ComputeTaskData &data, core::CommandBuffer &commandBuffer) {
                auto actual = data.output->actual();
                commandBuffer.bind_compute_pipeline_layout(ShaderManager::get_singleton().load_shader("base/compute/atomic_counter.comp",
                                                                                                      VK_SHADER_STAGE_COMPUTE_BIT));
                commandBuffer.bind_buffer(*actual, 0, actual->get_size(), 0, 6, 0);
                commandBuffer.dispatch(1);
            });
    }

    scene->play();
    return main_camera;
}

}// namespace vox
