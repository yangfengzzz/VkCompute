//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "compute/compute_pass.h"
#include "core/device.h"

#include <utility>

namespace vox::compute {

ComputePass::ComputePass(std::shared_ptr<ShaderSource> cs_source)
    : cs_source{std::move(cs_source)} {
}

void ComputePass::attach_shader_data(ShaderData *data) {
    auto iter = std::find(data_.begin(), data_.end(), data);
    if (iter == data_.end()) {
        data_.push_back(data);
    } else {
        LOGE("ShaderData already attached.")
    }
}

void ComputePass::detach_shader_data(ShaderData *data) {
    auto iter = std::find(data_.begin(), data_.end(), data);
    if (iter != data_.end()) {
        data_.erase(iter);
    }
}

void ComputePass::prepare(core::CommandBuffer &command_buffer) {
}

void ComputePass::compute(core::CommandBuffer &command_buffer) {
    ShaderVariant cs_variant;
    for (const auto &data : data_) {
        data->merge_variants(cs_variant, cs_variant);
    }

    // Get compute shader from cache
    auto &resource_cache = command_buffer.get_device().get_resource_cache();
    auto &shader_module = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, *cs_source, cs_variant);

    // Create pipeline layout and bind it
    auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});
    command_buffer.bind_pipeline_layout(pipeline_layout);

    auto &bindings = pipeline_layout.get_descriptor_set_layout(0);
    // Bind samplers to set = 0, binding = <according to name>
    for (const auto &data : data_) {
        data->bind_data(command_buffer, bindings);
    }

    if (!push_constants_data.empty()) {
        command_buffer.push_constants(push_constants_data);
    }

    // Dispatch compute
    command_buffer.dispatch(n_workgroups[0], n_workgroups[1], n_workgroups[2]);
}

}// namespace vox::compute