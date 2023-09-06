//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "ecs/components_manager.h"
#include "graphics_application.h"
#include "light/light_manager.h"
#include "mesh/mesh_manager.h"
#include "ecs/scene_manager.h"
#include "shader/shader_manager.h"
#include "texture/texture_manager.h"
#include "framework/fg/framegraph.h"
#include "framework/fg/vk_resource.h"

namespace vox {
class ForwardApplication : public GraphicsApplication {
public:
    fg::Framegraph framegraph;

    ForwardApplication() = default;

    ~ForwardApplication() override;

    /**
     * @brief Additional sample initialization
     */
    bool prepare(const ApplicationOptions &options) final;

    /**
     * @brief Main loop sample events
     */
    void update(float delta_time) override;

    bool resize(uint32_t win_width, uint32_t win_height, uint32_t fb_width, uint32_t fb_height) override;

    void input_event(const vox::InputEvent &input_event) override;

    void render(core::CommandBuffer &command_buffer) override;

    virtual void before_prepare() {}

    virtual void after_prepare() {}

    virtual Camera *load_scene() = 0;

    virtual void after_load_scene() {}

private:
    Camera *main_camera_{nullptr};

    /**
     * @brief Holds all scene information
     */
    std::unique_ptr<TextureManager> texture_manager_{nullptr};
    std::unique_ptr<ShaderManager> shader_manager_{nullptr};
    std::unique_ptr<MeshManager> mesh_manager_{nullptr};

    std::unique_ptr<ComponentsManager> components_manager_{nullptr};
    std::unique_ptr<SceneManager> scene_manager_{nullptr};
    std::unique_ptr<LightManager> light_manager_{nullptr};
};

}// namespace vox
