//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "components/renderer.h"

namespace vox {
class MeshRenderer : public Renderer {
public:
    using MeshPtr = std::shared_ptr<Mesh>;

    explicit MeshRenderer(Entity *entity);

    /**
     * Mesh assigned to the renderer.
     */
    void set_mesh(const MeshPtr &mesh);

    MeshPtr get_mesh();

private:
    void render(std::vector<RenderElement> &opaque_queue,
                std::vector<RenderElement> &alpha_test_queue,
                std::vector<RenderElement> &transparent_queue) override;

    void update_bounds(BoundingBox3F &world_bounds) override;

private:
    MeshPtr mesh_;
    std::unique_ptr<UpdateFlag> mesh_update_flag_;
};

}// namespace vox
