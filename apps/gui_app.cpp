//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "apps/gui_app.h"

#include "application/components/camera.h"
#include "application/components/mesh_renderer.h"
#include "application/controls/orbit_control.h"
#include "application/ecs/entity.h"
#include "application/material/unlit_material.h"
#include "application/mesh/primitive_mesh.h"

namespace vox {
namespace {
class GUIScript : public Script {
public:
    explicit GUIScript(Entity *entity) : Script(entity) {}

    void on_update(float delta_time) override {
        bool show_demo_window = true;
        ImGui::ShowDemoWindow(&show_demo_window);
    }
};

}// namespace

Camera *GUIApp::load_scene() {
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
    point_light->intensity_ = 1.0;
    point_light->distance_ = 100;

    auto cube_entity = root_entity->create_child();
    cube_entity->add_component<GUIScript>();
    auto renderer = cube_entity->add_component<MeshRenderer>();
    renderer->set_mesh(PrimitiveMesh::create_cuboid(1));
    auto material = std::make_shared<UnlitMaterial>(*device);
    material->set_base_color(Color(0.4, 0.6, 0.6));
    renderer->set_material(material);

    scene->play();
    return main_camera;
}

}// namespace vox
