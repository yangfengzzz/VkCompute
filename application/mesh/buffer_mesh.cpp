//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mesh/buffer_mesh.h"

namespace vox {
void BufferMesh::set_vertex_buffer_binding(size_t index, core::Buffer *binding) {
    if (vertex_buffer_bindings_.size() > index) {
        vertex_buffer_bindings_[index] = binding;
    } else {
        vertex_buffer_bindings_.insert(vertex_buffer_bindings_.begin() + index, binding);
    }
}

size_t BufferMesh::get_vertex_buffer_count() const { return vertex_buffer_bindings_.size(); }

const core::Buffer *BufferMesh::get_vertex_buffer(size_t index) const { return vertex_buffer_bindings_[index]; }

}  // namespace vox