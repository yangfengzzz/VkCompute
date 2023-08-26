//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "postprocessing_computepass.h"

#include <utility>

#include "postprocessing_pipeline.h"

namespace vox::rendering {

PostProcessingComputePass::PostProcessingComputePass(PostProcessingPipeline *parent,
                                                     std::shared_ptr<ShaderModule> cs_source)
    : PostProcessingPass{parent},
      cs_source{std::move(cs_source)} {
}

void PostProcessingComputePass::attach_shader_data(ShaderData *data) {
    auto iter = std::find(data_.begin(), data_.end(), data);
    if (iter == data_.end()) {
        data_.push_back(data);
    } else {
        LOGE("ShaderData already attached.")
    }
}

void PostProcessingComputePass::detach_shader_data(ShaderData *data) {
    auto iter = std::find(data_.begin(), data_.end(), data);
    if (iter != data_.end()) {
        data_.erase(iter);
    }
}

void PostProcessingComputePass::prepare(core::CommandBuffer &command_buffer, RenderTarget &default_render_target) {
}

void PostProcessingComputePass::draw(core::CommandBuffer &command_buffer, RenderTarget &default_render_target) {
    // Get cache
    auto &resource_cache = command_buffer.get_device().get_resource_cache();

    // Create pipeline layout and bind it
    auto &pipeline_layout = resource_cache.request_pipeline_layout({cs_source.get()});
    command_buffer.bind_pipeline_layout(pipeline_layout);

    auto &bindings = pipeline_layout.get_descriptor_set_layout(0);
    // Bind samplers to set = 0, binding = <according to name>
    for (const auto &data : data_) {
        data->bind_data(command_buffer, bindings);
        data->bind_specialization_constant(command_buffer, *cs_source);
    }

    if (!push_constants_data.empty()) {
        command_buffer.push_constants(push_constants_data);
    }

    // Dispatch compute
    command_buffer.dispatch(n_workgroups[0], n_workgroups[1], n_workgroups[2]);
}

PostProcessingComputePass::BarrierInfo PostProcessingComputePass::get_src_barrier_info() const {
    BarrierInfo info{};
    info.pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.image_read_access = VK_ACCESS_SHADER_READ_BIT;
    info.image_write_access = VK_ACCESS_SHADER_WRITE_BIT;
    return info;
}

PostProcessingComputePass::BarrierInfo PostProcessingComputePass::get_dst_barrier_info() const {
    BarrierInfo info{};
    info.pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.image_read_access = VK_ACCESS_SHADER_READ_BIT;
    info.image_write_access = VK_ACCESS_SHADER_WRITE_BIT;
    return info;
}

}// namespace vox::rendering