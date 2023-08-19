//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "subpass.h"

#include <utility>

#include "render_context.h"

namespace vox {
namespace rendering {

Subpass::Subpass(RenderContext &render_context, ShaderSource &&vertex_source,
                 ShaderSource &&fragment_source) : render_context{render_context},
                                                   vertex_shader{std::move(vertex_source)},
                                                   fragment_shader{std::move(fragment_source)} {
}

void Subpass::update_render_target_attachments(RenderTarget &render_target) {
    render_target.set_input_attachments(input_attachments);
    render_target.set_output_attachments(output_attachments);
}

RenderContext &Subpass::get_render_context() {
    return render_context;
}

const ShaderSource &Subpass::get_vertex_shader() const {
    return vertex_shader;
}

const ShaderSource &Subpass::get_fragment_shader() const {
    return fragment_shader;
}

core::DepthStencilState &Subpass::get_depth_stencil_state() {
    return depth_stencil_state;
}

const std::vector<uint32_t> &Subpass::get_input_attachments() const {
    return input_attachments;
}

void Subpass::set_input_attachments(std::vector<uint32_t> input) {
    input_attachments = std::move(input);
}

const std::vector<uint32_t> &Subpass::get_output_attachments() const {
    return output_attachments;
}

void Subpass::set_output_attachments(std::vector<uint32_t> output) {
    output_attachments = std::move(output);
}

const std::vector<uint32_t> &Subpass::get_color_resolve_attachments() const {
    return color_resolve_attachments;
}

void Subpass::set_color_resolve_attachments(std::vector<uint32_t> color_resolve) {
    color_resolve_attachments = std::move(color_resolve);
}

const bool &Subpass::get_disable_depth_stencil_attachment() const {
    return disable_depth_stencil_attachment;
}

void Subpass::set_disable_depth_stencil_attachment(bool disable_depth_stencil) {
    disable_depth_stencil_attachment = disable_depth_stencil;
}

const uint32_t &Subpass::get_depth_stencil_resolve_attachment() const {
    return depth_stencil_resolve_attachment;
}

void Subpass::set_depth_stencil_resolve_attachment(uint32_t depth_stencil_resolve) {
    depth_stencil_resolve_attachment = depth_stencil_resolve;
}

VkResolveModeFlagBits Subpass::get_depth_stencil_resolve_mode() const {
    return depth_stencil_resolve_mode;
}

void Subpass::set_depth_stencil_resolve_mode(VkResolveModeFlagBits mode) {
    depth_stencil_resolve_mode = mode;
}

void Subpass::set_sample_count(VkSampleCountFlagBits sample_count) {
    this->sample_count = sample_count;
}

const std::string &Subpass::get_debug_name() const {
    return debug_name;
}

void Subpass::set_debug_name(const std::string &name) {
    debug_name = name;
}

}
}// namespace vox::rendering
