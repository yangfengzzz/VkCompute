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
private:
    const std::string atomic_prop_;
    std::unique_ptr<core::Buffer> atomic_buffer_{nullptr};

public:
    explicit AtomicMaterial(core::Device &device) : BaseMaterial(device, "atomicRender"), atomic_prop_("atomicCounter") {
        atomic_buffer_ = std::make_unique<core::Buffer>(device, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                        VMA_MEMORY_USAGE_GPU_ONLY);
        shader_data_.set_buffer_functor(atomic_prop_, [this]() -> core::Buffer * { return atomic_buffer_.get(); });

        vertex_source_ = ShaderManager::get_singleton().load_shader("base/unlit.vert");
        fragment_source_ = ShaderManager::get_singleton().load_shader("base/compute/atomic_counter.frag");
    }
};

// MARK: - AtomicComputeApp
void AtomicComputeApp::load_scene() {
    auto scene = scene_manager_->get_current_scene();
    auto root_entity = scene->create_root_entity();

    auto camera_entity = root_entity->create_child();
    camera_entity->transform->set_position(10, 10, 10);
    camera_entity->transform->look_at(Point3F(0, 0, 0));
    main_camera_ = camera_entity->add_component<Camera>();
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

    scene->play();
}

bool AtomicComputeApp::prepare(const ApplicationOptions &options) {
    ForwardApplication::prepare(options);

    pipeline_ = std::make_unique<rendering::PostProcessingPipeline>(*render_context, ShaderSource());
    auto atomic_pass = &pipeline_->add_pass<rendering::PostProcessingComputePass>(
        ShaderManager::get_singleton().load_shader("base/compute/atomic_counter.comp"));
    atomic_pass->set_dispatch_size({1, 1, 1});
    atomic_pass->attach_shader_data(&material_->shader_data_);

    return true;
}

void AtomicComputeApp::update_gpu_task(core::CommandBuffer &command_buffer) {
    ForwardApplication::update_gpu_task(command_buffer);
    pipeline_->draw(command_buffer);
}

}// namespace vox
