//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "framework/rendering/subpass.h"
#include "subpasses/render_element.h"

namespace vox {
class Scene;
class Camera;

/**
 * @brief This subpass is responsible for rendering a Scene
 */
class GeometrySubpass : public rendering::Subpass {
public:
    /**
     * @brief Constructs a subpass for the geometry pass of Deferred rendering
     * @param render_context Render context
     * @param scene Scene to render on this subpass
     * @param camera Camera used to look at the scene
     */
    GeometrySubpass(rendering::RenderContext &render_context, Scene *scene, Camera *camera);

    ~GeometrySubpass() override = default;

    void prepare() override;

    /**
     * @brief Record draw commands
     */
    void draw(core::CommandBuffer &command_buffer) override;

    /**
     * @brief Thread index to use for allocating resources
     */
    void set_thread_index(uint32_t index);

protected:
    void draw_element(core::CommandBuffer &command_buffer,
                      const std::vector<RenderElement> &items,
                      const ShaderVariant &variant);

    uint32_t thread_index_{0};

    Scene *scene_{nullptr};
    Camera *camera_{nullptr};

    // A map of shader resource names and the mode of constant data
    std::unordered_map<std::string, ShaderResourceMode> resource_mode_map_;

    static bool compare_from_near_to_far(const RenderElement &a, const RenderElement &b);

    static bool compare_from_far_to_near(const RenderElement &a, const RenderElement &b);

    virtual core::PipelineLayout &prepare_pipeline_layout(core::CommandBuffer &command_buffer,
                                                          const std::vector<ShaderModule *> &shader_modules);
};

}// namespace vox
