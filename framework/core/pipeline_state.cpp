//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "pipeline_state.h"

bool operator==(const VkVertexInputAttributeDescription &lhs, const VkVertexInputAttributeDescription &rhs) {
    return std::tie(lhs.binding, lhs.format, lhs.location, lhs.offset) == std::tie(rhs.binding, rhs.format, rhs.location, rhs.offset);
}

bool operator==(const VkVertexInputBindingDescription &lhs, const VkVertexInputBindingDescription &rhs) {
    return std::tie(lhs.binding, lhs.inputRate, lhs.stride) == std::tie(rhs.binding, rhs.inputRate, rhs.stride);
}

bool operator==(const vox::core::ColorBlendAttachmentState &lhs, const vox::core::ColorBlendAttachmentState &rhs) {
    return std::tie(lhs.alpha_blend_op, lhs.blend_enable, lhs.color_blend_op, lhs.color_write_mask,
                    lhs.dst_alpha_blend_factor, lhs.dst_color_blend_factor, lhs.src_alpha_blend_factor, lhs.src_color_blend_factor) ==
           std::tie(rhs.alpha_blend_op, rhs.blend_enable, rhs.color_blend_op, rhs.color_write_mask,
                    rhs.dst_alpha_blend_factor, rhs.dst_color_blend_factor, rhs.src_alpha_blend_factor, rhs.src_color_blend_factor);
}

bool operator!=(const vox::core::StencilOpState &lhs, const vox::core::StencilOpState &rhs) {
    return std::tie(lhs.compare_op, lhs.depth_fail_op, lhs.fail_op, lhs.pass_op) != std::tie(rhs.compare_op, rhs.depth_fail_op, rhs.fail_op, rhs.pass_op);
}

bool operator!=(const vox::core::VertexInputState &lhs, const vox::core::VertexInputState &rhs) {
    return lhs.attributes != rhs.attributes || lhs.bindings != rhs.bindings;
}

bool operator!=(const vox::core::InputAssemblyState &lhs, const vox::core::InputAssemblyState &rhs) {
    return std::tie(lhs.primitive_restart_enable, lhs.topology) != std::tie(rhs.primitive_restart_enable, rhs.topology);
}

bool operator!=(const vox::core::RasterizationState &lhs, const vox::core::RasterizationState &rhs) {
    return std::tie(lhs.cull_mode, lhs.depth_bias_enable, lhs.depth_clamp_enable,
                    lhs.front_face, lhs.front_face, lhs.polygon_mode, lhs.rasterizer_discard_enable) !=
           std::tie(rhs.cull_mode, rhs.depth_bias_enable, rhs.depth_clamp_enable,
                    rhs.front_face, rhs.front_face, rhs.polygon_mode, rhs.rasterizer_discard_enable);
}

bool operator!=(const vox::core::ViewportState &lhs, const vox::core::ViewportState &rhs) {
    return lhs.viewport_count != rhs.viewport_count || lhs.scissor_count != rhs.scissor_count;
}

bool operator!=(const vox::core::MultisampleState &lhs, const vox::core::MultisampleState &rhs) {
    return std::tie(lhs.alpha_to_coverage_enable, lhs.alpha_to_one_enable, lhs.min_sample_shading,
                    lhs.rasterization_samples, lhs.sample_mask, lhs.sample_shading_enable) !=
           std::tie(rhs.alpha_to_coverage_enable, rhs.alpha_to_one_enable, rhs.min_sample_shading,
                    rhs.rasterization_samples, rhs.sample_mask, rhs.sample_shading_enable);
}

bool operator!=(const vox::core::DepthStencilState &lhs, const vox::core::DepthStencilState &rhs) {
    return std::tie(lhs.depth_bounds_test_enable, lhs.depth_compare_op, lhs.depth_test_enable, lhs.depth_write_enable, lhs.stencil_test_enable) !=
               std::tie(rhs.depth_bounds_test_enable, rhs.depth_compare_op, rhs.depth_test_enable, rhs.depth_write_enable, rhs.stencil_test_enable) ||
           lhs.back != rhs.back || lhs.front != rhs.front;
}

bool operator!=(const vox::core::ColorBlendState &lhs, const vox::core::ColorBlendState &rhs) {
    return std::tie(lhs.logic_op, lhs.logic_op_enable) != std::tie(rhs.logic_op, rhs.logic_op_enable) ||
           lhs.attachments.size() != rhs.attachments.size() ||
           !std::equal(lhs.attachments.begin(), lhs.attachments.end(), rhs.attachments.begin(),
                       [](const vox::core::ColorBlendAttachmentState &lhs, const vox::core::ColorBlendAttachmentState &rhs) {
                           return lhs == rhs;
                       });
}

namespace vox::core {
void SpecializationConstantState::reset() {
    if (dirty) {
        specialization_constant_state.clear();
    }

    dirty = false;
}

bool SpecializationConstantState::is_dirty() const {
    return dirty;
}

void SpecializationConstantState::clear_dirty() {
    dirty = false;
}

void SpecializationConstantState::set_constant(uint32_t constant_id, const std::vector<uint8_t> &value) {
    auto data = specialization_constant_state.find(constant_id);

    if (data != specialization_constant_state.end() && data->second == value) {
        return;
    }

    dirty = true;

    specialization_constant_state[constant_id] = value;
}

void SpecializationConstantState::set_specialization_constant_state(const std::map<uint32_t, std::vector<uint8_t>> &state) {
    specialization_constant_state = state;
}

const std::map<uint32_t, std::vector<uint8_t>> &SpecializationConstantState::get_specialization_constant_state() const {
    return specialization_constant_state;
}

void PipelineState::reset() {
    clear_dirty();

    pipeline_layout = nullptr;

    render_pass = nullptr;

    specialization_constant_state.reset();

    vertex_input_state = {};

    input_assembly_state = {};

    rasterization_state = {};

    multisample_state = {};

    depth_stencil_state = {};

    color_blend_state = {};

    subpass_index = {0U};
}

void PipelineState::set_pipeline_layout(PipelineLayout &new_pipeline_layout) {
    if (pipeline_layout) {
        if (pipeline_layout->get_handle() != new_pipeline_layout.get_handle()) {
            pipeline_layout = &new_pipeline_layout;

            dirty = true;
        }
    } else {
        pipeline_layout = &new_pipeline_layout;

        dirty = true;
    }
}

void PipelineState::set_render_pass(const RenderPass &new_render_pass) {
    if (render_pass) {
        if (render_pass->get_handle() != new_render_pass.get_handle()) {
            render_pass = &new_render_pass;

            dirty = true;
        }
    } else {
        render_pass = &new_render_pass;

        dirty = true;
    }
}

void PipelineState::set_specialization_constant(uint32_t constant_id, const std::vector<uint8_t> &data) {
    specialization_constant_state.set_constant(constant_id, data);

    if (specialization_constant_state.is_dirty()) {
        dirty = true;
    }
}

void PipelineState::set_vertex_input_state(const VertexInputState &new_vertex_input_state) {
    if (vertex_input_state != new_vertex_input_state) {
        vertex_input_state = new_vertex_input_state;

        dirty = true;
    }
}

void PipelineState::set_input_assembly_state(const InputAssemblyState &new_input_assembly_state) {
    if (input_assembly_state != new_input_assembly_state) {
        input_assembly_state = new_input_assembly_state;

        dirty = true;
    }
}

void PipelineState::set_rasterization_state(const RasterizationState &new_rasterization_state) {
    if (rasterization_state != new_rasterization_state) {
        rasterization_state = new_rasterization_state;

        dirty = true;
    }
}

void PipelineState::set_viewport_state(const ViewportState &new_viewport_state) {
    if (viewport_state != new_viewport_state) {
        viewport_state = new_viewport_state;

        dirty = true;
    }
}

void PipelineState::set_multisample_state(const MultisampleState &new_multisample_state) {
    if (multisample_state != new_multisample_state) {
        multisample_state = new_multisample_state;

        dirty = true;
    }
}

void PipelineState::set_depth_stencil_state(const DepthStencilState &new_depth_stencil_state) {
    if (depth_stencil_state != new_depth_stencil_state) {
        depth_stencil_state = new_depth_stencil_state;

        dirty = true;
    }
}

void PipelineState::set_color_blend_state(const ColorBlendState &new_color_blend_state) {
    if (color_blend_state != new_color_blend_state) {
        color_blend_state = new_color_blend_state;

        dirty = true;
    }
}

void PipelineState::set_subpass_index(uint32_t new_subpass_index) {
    if (subpass_index != new_subpass_index) {
        subpass_index = new_subpass_index;

        dirty = true;
    }
}

const PipelineLayout &PipelineState::get_pipeline_layout() const {
    assert(pipeline_layout && "Graphics state Pipeline layout is not set");
    return *pipeline_layout;
}

const RenderPass *PipelineState::get_render_pass() const {
    return render_pass;
}

const SpecializationConstantState &PipelineState::get_specialization_constant_state() const {
    return specialization_constant_state;
}

const VertexInputState &PipelineState::get_vertex_input_state() const {
    return vertex_input_state;
}

const InputAssemblyState &PipelineState::get_input_assembly_state() const {
    return input_assembly_state;
}

const RasterizationState &PipelineState::get_rasterization_state() const {
    return rasterization_state;
}

const ViewportState &PipelineState::get_viewport_state() const {
    return viewport_state;
}

const MultisampleState &PipelineState::get_multisample_state() const {
    return multisample_state;
}

const DepthStencilState &PipelineState::get_depth_stencil_state() const {
    return depth_stencil_state;
}

const ColorBlendState &PipelineState::get_color_blend_state() const {
    return color_blend_state;
}

uint32_t PipelineState::get_subpass_index() const {
    return subpass_index;
}

bool PipelineState::is_dirty() const {
    return dirty || specialization_constant_state.is_dirty();
}

void PipelineState::clear_dirty() {
    dirty = false;
    specialization_constant_state.clear_dirty();
}

}// namespace vox::core
