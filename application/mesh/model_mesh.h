//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/color.h"
#include "mesh/mesh.h"

namespace vox {
struct ValueChanged {
    enum Enum {
        POSITION = 0x1,
        NORMAL = 0x2,
        COLOR = 0x4,
        TANGENT = 0x8,
        BONE_WEIGHT = 0x10,
        BONE_INDEX = 0x20,
        UV = 0x40,
        UV_1 = 0x80,
        UV_2 = 0x100,
        UV_3 = 0x200,
        UV_4 = 0x400,
        UV_5 = 0x800,
        UV_6 = 0x1000,
        UV_7 = 0x2000,
        BLEND_SHAPE = 0x4000,
        ALL = 0xffff
    };
};

/**
 * Mesh containing common vertex elements of the model.
 */
class ModelMesh : public Mesh {
public:
    /**
     * Whether to access data of the mesh.
     */
    [[nodiscard]] bool is_accessible() const;

    /**
     * Vertex count of current mesh.
     */
    [[nodiscard]] size_t get_vertex_count() const;

    /**
     * Create a model mesh.
     * @param device - Engine to which the mesh belongs
     */
    explicit ModelMesh(core::Device &device);

    /**
     * Vertex buffer binding collection.
     */
    [[nodiscard]] const std::vector<std::unique_ptr<core::Buffer>> &get_vertex_buffer_bindings() const;

    void set_vertex_buffer_binding(size_t index, std::unique_ptr<core::Buffer> &&binding);

    [[nodiscard]] size_t get_vertex_buffer_count() const override;

    [[nodiscard]] const core::Buffer *get_vertex_buffer(size_t index) const override;

public:
    /**
     * Set positions for the mesh.
     * @param positions - The positions for the mesh.
     */
    void set_positions(const std::vector<Vector3F> &positions);

    /**
     * Get positions for the mesh.
     * @remarks Please call the SetPositions() method after modification to ensure that the modification takes effect.
     */
    const std::vector<Vector3F> &get_positions();

    /**
     * Set per-vertex normals for the mesh.
     * @param normals - The normals for the mesh.
     */
    void set_normals(const std::vector<Vector3F> &normals);

    /**
     * Get normals for the mesh.
     * @remarks Please call the SetNormals() method after modification to ensure that the modification takes effect.
     */
    const std::vector<Vector3F> &get_normals();

    /**
     * Set per-vertex colors for the mesh.
     * @param colors - The colors for the mesh.
     */
    void set_colors(const std::vector<Color> &colors);

    /**
     * Get colors for the mesh.
     * @remarks Please call the SetColors() method after modification to ensure that the modification takes effect.
     */
    const std::vector<Color> &get_colors();

    /**
     * Set per-vertex tangents for the mesh.
     * @param tangents - The tangents for the mesh.
     */
    void set_tangents(const std::vector<Vector4F> &tangents);

    /**
     * Get tangents for the mesh.
     * @remarks Please call the SetTangents() method after modification to ensure that the modification takes effect.
     */
    const std::vector<Vector4F> &get_tangents();

    /**
     * Set per-vertex uv for the mesh by channelIndex.
     * @param uv - The uv for the mesh.
     * @param channel_index - The index of uv channels, in [0 ~ 7] range.
     */
    void set_uvs(const std::vector<Vector2F> &uv, int channel_index = 0);

    /**
     * Get uv for the mesh by channelIndex.
     * @param channel_index - The index of uv channels, in [0 ~ 7] range.
     * @remarks Please call the setUV() method after modification to ensure that the modification takes effect.
     */
    const std::vector<Vector2F> &get_uvs(int channel_index = 0);

    /**
     * Set indices for the mesh.
     * @param indices - The indices for the mesh.
     */
    void set_indices(const std::vector<uint32_t> &indices);

    /**
     * Set indices for the mesh.
     * @param indices - The indices for the mesh.
     */
    void set_indices(const std::vector<uint16_t> &indices);

    /**
     * Upload Mesh Data to the graphics API.
     * @param no_longer_accessible - Whether to access data later. If true, you'll never access data anymore (free
     * memory cache)
     */
    void upload_data(bool no_longer_accessible);

private:
    static inline VkVertexInputAttributeDescription vertex_input_attribute_description(
        uint32_t binding,
        uint32_t location,
        VkFormat format,
        uint32_t offset) {
        VkVertexInputAttributeDescription vertex_input_attribute_description{};
        vertex_input_attribute_description.location = location;
        vertex_input_attribute_description.binding = binding;
        vertex_input_attribute_description.format = format;
        vertex_input_attribute_description.offset = offset;
        return vertex_input_attribute_description;
    }

    static inline VkVertexInputBindingDescription vertex_input_binding_description(
        uint32_t binding,
        uint32_t stride,
        VkVertexInputRate input_rate) {
        VkVertexInputBindingDescription vertex_input_binding_description{};
        vertex_input_binding_description.binding = binding;
        vertex_input_binding_description.stride = stride;
        vertex_input_binding_description.inputRate = input_rate;
        return vertex_input_binding_description;
    }

    core::Device &device_;
    std::vector<std::unique_ptr<core::Buffer>> vertex_buffer_bindings_{};

    void update_vertex_state();

    void update_vertices(std::vector<float> &vertices);

    void release_cache();

    bool has_blend_shape_ = false;
    bool use_blend_shape_normal_ = false;
    bool use_blend_shape_tangent_ = false;

    size_t vertex_count_ = 0;
    bool accessible_ = true;
    std::vector<float> vertices_{};
    std::vector<uint32_t> indices_32_{};
    std::vector<uint16_t> indices_16_{};
    VkIndexType indices_type_;
    int vertex_change_flag_{};
    size_t element_count_{};

    std::vector<Vector3F> positions_;
    std::vector<Vector3F> normals_;
    std::vector<Color> colors_;
    std::vector<Vector4F> tangents_;
    std::vector<Vector2F> uv_;
    std::vector<Vector2F> uv_1_;
    std::vector<Vector2F> uv_2_;
    std::vector<Vector2F> uv_3_;
    std::vector<Vector2F> uv_4_;
    std::vector<Vector2F> uv_5_;
    std::vector<Vector2F> uv_6_;
    std::vector<Vector2F> uv_7_;
    std::vector<Vector4F> bone_weights_;
    std::vector<Vector4F> bone_indices_;
};
using ModelMeshPtr = std::shared_ptr<ModelMesh>;

}// namespace vox
