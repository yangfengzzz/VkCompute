//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mesh/model_mesh.h"

#include "framework/core/device.h"
#include "mesh/mesh_attributes.h"

namespace vox {
bool ModelMesh::is_accessible() const { return accessible_; }

size_t ModelMesh::get_vertex_count() const { return vertex_count_; }

ModelMesh::ModelMesh(core::Device &device) : device_(device) {}

void ModelMesh::set_vertex_buffer_binding(size_t index, std::unique_ptr<core::Buffer> &&binding) {
    if (vertex_buffer_bindings_.size() > index) {
        vertex_buffer_bindings_[index] = std::move(binding);
    } else {
        vertex_buffer_bindings_.insert(vertex_buffer_bindings_.begin() + index, std::move(binding));
    }
}

const std::vector<std::unique_ptr<core::Buffer>> &ModelMesh::get_vertex_buffer_bindings() const {
    return vertex_buffer_bindings_;
}

size_t ModelMesh::get_vertex_buffer_count() const { return vertex_buffer_bindings_.size(); }

const core::Buffer *ModelMesh::get_vertex_buffer(size_t index) const { return vertex_buffer_bindings_[index].get(); }

void ModelMesh::set_positions(const std::vector<Vector3F> &positions) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    auto count = positions.size();
    positions_ = positions;
    vertex_change_flag_ |= ValueChanged::POSITION;

    if (vertex_count_ != count) {
        vertex_count_ = count;
    }
}

const std::vector<Vector3F> &ModelMesh::get_positions() {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    return positions_;
}

void ModelMesh::set_normals(const std::vector<Vector3F> &normals) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    if (normals.size() != vertex_count_) {
        assert(false && "The array provided needs to be the same size as vertex count.");
    }

    vertex_change_flag_ |= ValueChanged::NORMAL;
    normals_ = normals;
}

const std::vector<Vector3F> &ModelMesh::get_normals() {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }
    return normals_;
}

void ModelMesh::set_colors(const std::vector<Color> &colors) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    if (colors.size() != vertex_count_) {
        assert(false && "The array provided needs to be the same size as vertex count.");
    }

    vertex_change_flag_ |= ValueChanged::COLOR;
    colors_ = colors;
}

const std::vector<Color> &ModelMesh::get_colors() {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }
    return colors_;
}

void ModelMesh::set_tangents(const std::vector<Vector4F> &tangents) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    if (tangents.size() != vertex_count_) {
        assert(false && "The array provided needs to be the same size as vertex count.");
    }

    vertex_change_flag_ |= ValueChanged::TANGENT;
    tangents_ = tangents;
}

const std::vector<Vector4F> &ModelMesh::get_tangents() {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }
    return tangents_;
}

void ModelMesh::set_uvs(const std::vector<Vector2F> &uv, int channel_index) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    if (uv.size() != vertex_count_) {
        assert(false && "The array provided needs to be the same size as vertex count.");
    }

    switch (channel_index) {
        case 0:
            vertex_change_flag_ |= ValueChanged::UV;
            uv_ = uv;
            break;
        case 1:
            vertex_change_flag_ |= ValueChanged::UV_1;
            uv_1_ = uv;
            break;
        case 2:
            vertex_change_flag_ |= ValueChanged::UV_2;
            uv_2_ = uv;
            break;
        case 3:
            vertex_change_flag_ |= ValueChanged::UV_3;
            uv_3_ = uv;
            break;
        case 4:
            vertex_change_flag_ |= ValueChanged::UV_4;
            uv_4_ = uv;
            break;
        case 5:
            vertex_change_flag_ |= ValueChanged::UV_5;
            uv_5_ = uv;
            break;
        case 6:
            vertex_change_flag_ |= ValueChanged::UV_6;
            uv_6_ = uv;
            break;
        case 7:
            vertex_change_flag_ |= ValueChanged::UV_7;
            uv_7_ = uv;
            break;
        default:
            assert(false && "The index of channel needs to be in range [0 - 7].");
    }
}

const std::vector<Vector2F> &ModelMesh::get_uvs(int channel_index) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }
    switch (channel_index) {
        case 0:
            return uv_;
        case 1:
            return uv_1_;
        case 2:
            return uv_2_;
        case 3:
            return uv_3_;
        case 4:
            return uv_4_;
        case 5:
            return uv_5_;
        case 6:
            return uv_6_;
        case 7:
            return uv_7_;
        default:
            assert(false && "The index of channel needs to be in range [0 - 7].");
            throw std::exception();
    }
}

void ModelMesh::set_indices(const std::vector<uint32_t> &indices) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }
    indices_type_ = VkIndexType::VK_INDEX_TYPE_UINT32;
    indices_32_ = indices;
}

void ModelMesh::set_indices(const std::vector<uint16_t> &indices) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }
    indices_type_ = VkIndexType::VK_INDEX_TYPE_UINT16;
    indices_16_ = indices;
}

void ModelMesh::upload_data(bool no_longer_accessible) {
    if (!accessible_) {
        assert(false && "Not allowed to access data while accessible is false.");
    }

    update_vertex_state();
    vertex_change_flag_ = ValueChanged::ALL;

    auto vertex_float_count = element_count_ * vertex_count_;
    auto vertices = std::vector<float>(vertex_float_count);
    update_vertices(vertices);

    auto &queue = device_.get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0);

    // keep stage buffer alive until submit finish
    std::vector<core::Buffer> transient_buffers;
    auto &command_buffer = device_.request_command_buffer();

    command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    core::Buffer stage_buffer{device_,
                              core::BufferDesc{.size = vertices.size() * sizeof(float),
                                               .buffer_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               .memory_usage = VMA_MEMORY_USAGE_CPU_ONLY}};

    stage_buffer.update(vertices.data(), vertices.size() * sizeof(float));

    auto new_vertex_buffer = std::make_unique<core::Buffer>(
        device_, core::BufferDesc{.size = vertices.size() * sizeof(float),
                                  .buffer_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                  .memory_usage = VMA_MEMORY_USAGE_GPU_ONLY});

    command_buffer.copy_buffer(stage_buffer, *new_vertex_buffer, vertices.size() * sizeof(float));
    set_vertex_buffer_binding(0, std::move(new_vertex_buffer));
    transient_buffers.push_back(std::move(stage_buffer));

    if (indices_type_ == VkIndexType::VK_INDEX_TYPE_UINT16) {
        core::Buffer stage_buffer{device_, core::BufferDesc{indices_16_.size() * sizeof(uint16_t),
                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                            VMA_MEMORY_USAGE_CPU_ONLY}};

        stage_buffer.update(indices_16_.data(), indices_16_.size() * sizeof(uint16_t));

        core::Buffer new_index_buffer{device_, core::BufferDesc{indices_16_.size() * sizeof(uint16_t),
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                VMA_MEMORY_USAGE_GPU_ONLY}};

        command_buffer.copy_buffer(stage_buffer, new_index_buffer, indices_16_.size() * sizeof(uint16_t));
        set_index_buffer_binding(std::make_unique<vox::IndexBufferBinding>(std::move(new_index_buffer), indices_type_));
        transient_buffers.push_back(std::move(stage_buffer));
    } else if (indices_type_ == VkIndexType::VK_INDEX_TYPE_UINT32) {
        core::Buffer stage_buffer{device_, core::BufferDesc{indices_32_.size() * sizeof(uint32_t),
                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                            VMA_MEMORY_USAGE_CPU_ONLY}};

        stage_buffer.update(indices_32_.data(), indices_32_.size() * sizeof(uint32_t));

        core::Buffer new_index_buffer{device_, core::BufferDesc{indices_32_.size() * sizeof(uint32_t),
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                VMA_MEMORY_USAGE_GPU_ONLY}};

        command_buffer.copy_buffer(stage_buffer, new_index_buffer, indices_32_.size() * sizeof(uint32_t));
        set_index_buffer_binding(std::make_unique<vox::IndexBufferBinding>(std::move(new_index_buffer), indices_type_));
        transient_buffers.push_back(std::move(stage_buffer));
    }

    command_buffer.end();

    queue.submit(command_buffer, device_.request_fence());

    device_.get_fence_pool().wait();
    device_.get_fence_pool().reset();
    device_.get_command_pool().reset_pool();

    if (no_longer_accessible) {
        accessible_ = false;
        release_cache();
    }
}

void ModelMesh::update_vertex_state() {
    auto &vertex_input_attributes = vertex_input_state_.attributes;
    vertex_input_attributes.resize(1);
    vertex_input_attributes[0] = vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);

    uint32_t offset = 12;
    uint32_t element_count = 3;
    if (!normals_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::NORMAL, VK_FORMAT_R32G32B32_SFLOAT, offset));
        offset += 12;
        element_count += 3;
    }
    if (!colors_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::COLOR_0, VK_FORMAT_R32G32B32A32_SFLOAT, offset));
        offset += 16;
        element_count += 4;
    }
    if (!tangents_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::TANGENT, VK_FORMAT_R32G32B32A32_SFLOAT, offset));
        offset += 16;
        element_count += 4;
    }
    if (!uv_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_0, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_1_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_1, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_2_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_2, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_3_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_3, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_4_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_4, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_5_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_5, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_6_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_6, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }
    if (!uv_7_.empty()) {
        vertex_input_attributes.push_back(vertex_input_attribute_description(
            0, (uint32_t)Attributes::UV_7, VK_FORMAT_R32G32_SFLOAT, offset));
        offset += 8;
        element_count += 2;
    }

    auto &vertex_input_bindings = vertex_input_state_.bindings;
    vertex_input_bindings.resize(1);
    vertex_input_bindings[0] = vertex_input_binding_description(0, element_count * 4, VK_VERTEX_INPUT_RATE_VERTEX);

    element_count_ = element_count;
}

void ModelMesh::update_vertices(std::vector<float> &vertices) {
    if ((vertex_change_flag_ & ValueChanged::POSITION) != 0) {
        for (size_t i = 0; i < vertex_count_; i++) {
            auto start = element_count_ * i;
            const auto &position = positions_[i];
            vertices[start] = position.x;
            vertices[start + 1] = position.y;
            vertices[start + 2] = position.z;
        }
    }

    size_t offset = 3;

    if (!normals_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::NORMAL) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &normal = normals_[i];
                vertices[start] = normal.x;
                vertices[start + 1] = normal.y;
                vertices[start + 2] = normal.z;
            }
        }
        offset += 3;
    }

    if (!colors_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::COLOR) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &color = colors_[i];
                vertices[start] = color.r;
                vertices[start + 1] = color.g;
                vertices[start + 2] = color.b;
                vertices[start + 3] = color.a;
            }
        }
        offset += 4;
    }

    if (!tangents_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::TANGENT) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &tangent = tangents_[i];
                vertices[start] = tangent.x;
                vertices[start + 1] = tangent.y;
                vertices[start + 2] = tangent.z;
            }
        }
        offset += 4;
    }
    if (!uv_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_1_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_1) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_1_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_2_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_2) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_2_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_3_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_3) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_3_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_4_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_4) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_4_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_5_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_5) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_5_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_6_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_6) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_6_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }
    if (!uv_7_.empty()) {
        if ((vertex_change_flag_ & ValueChanged::UV_7) != 0) {
            for (size_t i = 0; i < vertex_count_; i++) {
                auto start = element_count_ * i + offset;
                const auto &uv = uv_7_[i];
                vertices[start] = uv.x;
                vertices[start + 1] = uv.y;
            }
        }
        offset += 2;
    }

    vertex_change_flag_ = 0;
}

void ModelMesh::release_cache() {
    vertices_.clear();
    positions_.clear();
    tangents_.clear();
    normals_.clear();
    colors_.clear();
    uv_.clear();
    uv_1_.clear();
    uv_2_.clear();
    uv_3_.clear();
    uv_4_.clear();
    uv_5_.clear();
    uv_6_.clear();
    uv_7_.clear();
}

}// namespace vox
