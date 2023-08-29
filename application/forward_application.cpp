//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "forward_application.h"

#include "components/camera.h"
#include "framework/platform/platform.h"
#include "subpasses/geometry_subpass.h"

namespace vox {
ForwardApplication::~ForwardApplication() {
    // release first
    scene_manager_.reset();

    components_manager_.reset();
    light_manager_.reset();

    texture_manager_->collect_garbage();
    texture_manager_.reset();
    shader_manager_->collect_garbage();
    shader_manager_.reset();
    mesh_manager_->collect_garbage();
    mesh_manager_.reset();
}

bool ForwardApplication::prepare(const ApplicationOptions &options) {
    before_prepare();
    GraphicsApplication::prepare(options);
    after_prepare();

    // resource loader
    texture_manager_ = std::make_unique<TextureManager>(*device);
    shader_manager_ = std::make_unique<ShaderManager>(*device);
    mesh_manager_ = std::make_unique<MeshManager>(*device);

    // logic system
    components_manager_ = std::make_unique<ComponentsManager>();
    scene_manager_ = std::make_unique<SceneManager>(*device);
    auto scene = scene_manager_->get_current_scene();

    light_manager_ = std::make_unique<LightManager>(scene);
    {
        main_camera_ = load_scene();
        auto extent = options.window->get_extent();
        auto factor = static_cast<uint32_t>(options.window->get_content_scale_factor());
        components_manager_->call_script_resize(extent.width, extent.height, factor * extent.width,
                                                factor * extent.height);
        main_camera_->resize(extent.width, extent.height, factor * extent.width, factor * extent.height);
    }

    // default render pipeline
    std::vector<std::unique_ptr<rendering::Subpass>> scene_subpasses{};
    scene_subpasses.emplace_back(std::make_unique<GeometrySubpass>(get_render_context(), scene, main_camera_));
    set_render_pipeline(rendering::RenderPipeline(std::move(scene_subpasses)));

    after_load_scene();

    return true;
}

void ForwardApplication::update(float delta_time) {
    {
        components_manager_->call_script_on_start();

        components_manager_->call_script_on_update(delta_time);
        components_manager_->call_script_on_late_update(delta_time);

        components_manager_->call_renderer_on_update(delta_time);
        scene_manager_->get_current_scene()->update_shader_data();
    }

    GraphicsApplication::update(delta_time);
}

bool ForwardApplication::resize(uint32_t win_width, uint32_t win_height, uint32_t fb_width, uint32_t fb_height) {
    GraphicsApplication::resize(win_width, win_height, fb_width, fb_height);
    components_manager_->call_script_resize(win_width, win_height, fb_width, fb_height);
    main_camera_->resize(win_width, win_height, fb_width, fb_height);
    return true;
}

void ForwardApplication::input_event(const vox::InputEvent &input_event) {
    GraphicsApplication::input_event(input_event);
    components_manager_->call_script_input_event(input_event);
}

void ForwardApplication::render(core::CommandBuffer &command_buffer) {
    update_gpu_task(command_buffer);
    GraphicsApplication::render(command_buffer);
}

void ForwardApplication::update_gpu_task(core::CommandBuffer &command_buffer) {
}

}// namespace vox
