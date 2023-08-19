//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mesh/sub_mesh.h"

namespace vox {
SubMesh::SubMesh(uint32_t start, uint32_t count) : start_(start), count_(count) {}

uint32_t SubMesh::get_start() const { return start_; }

uint32_t SubMesh::get_count() const { return count_; }

}// namespace vox
