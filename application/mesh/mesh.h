//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

#include "base/update_flag_manager.h"
#include "math/bounding_box3.h"
#include "mesh/index_buffer_binding.h"
#include "mesh/sub_mesh.h"
#include "framework/core/pipeline_state.h"

namespace vox {
class Mesh {
public:
    /** Name. */
    std::string name_;
    /** The bounding volume of the mesh. */
    BoundingBox3F bounds_ = BoundingBox3F();

    /**
     * Instanced count, disable instanced drawing when set zero.
     */
    [[nodiscard]] uint32_t get_instance_count() const;

    void set_instance_count(uint32_t value);

    /**
     * First sub-mesh. Rendered using the first material.
     */
    [[nodiscard]] const SubMesh *get_first_sub_mesh() const;

    /**
     * A collection of sub-mesh, each sub-mesh can be rendered with an independent material.
     */
    [[nodiscard]] const std::vector<SubMesh> &get_sub_meshes() const;

    /**
     * Add sub-mesh, each sub-mesh can correspond to an independent material.
     * @param sub_mesh - Start drawing offset, if the index buffer is set,
     * it means the offset in the index buffer, if not set, it means the offset in the vertex buffer
     */
    void add_sub_mesh(SubMesh sub_mesh);

    /**
     * Add sub-mesh, each sub-mesh can correspond to an independent material.
     * @param start - Start drawing offset, if the index buffer is set,
     * it means the offset in the index buffer, if not set,
     * it means the offset in the vertex buffer
     * @param count - Drawing count, if the index buffer is set,
     * it means the count in the index buffer, if not set,
     * it means the count in the vertex buffer
     */
    void add_sub_mesh(uint32_t start = 0, uint32_t count = 0);

    /**
     * Clear all sub-mesh.
     */
    void clear_sub_mesh();

    /**
     * Register update flag, update flag will be true if the vertex element changes.
     * @returns Update flag
     */
    std::unique_ptr<UpdateFlag> register_update_flag();

public:
    /**
     * Set vertex state.
     * @param vertex_input_bindings - stride step size
     * @param vertex_input_attributes - Vertex attributes collection
     */
    void set_vertex_input_state(const std::vector<VkVertexInputBindingDescription> &vertex_input_bindings,
                                const std::vector<VkVertexInputAttributeDescription> &vertex_input_attributes);

    [[nodiscard]] const core::VertexInputState &get_vertex_input_state() const;

    /**
     * Index buffer binding.
     */
    [[nodiscard]] const IndexBufferBinding *get_index_buffer_binding() const;

    void set_index_buffer_binding(std::unique_ptr<vox::IndexBufferBinding> &&binding);

    [[nodiscard]] virtual size_t get_vertex_buffer_count() const = 0;

    [[nodiscard]] virtual const core::Buffer *get_vertex_buffer(size_t index) const = 0;

protected:
    uint32_t instance_count_ = 1;
    std::unique_ptr<vox::IndexBufferBinding> index_buffer_binding_{nullptr};
    core::VertexInputState vertex_input_state_;

    std::vector<SubMesh> sub_meshes_{};
    UpdateFlagManager update_flag_manager_;
};

}// namespace vox
