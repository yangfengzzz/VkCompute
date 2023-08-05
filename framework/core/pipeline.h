//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "rendering/pipeline_state.h"

namespace vox {
class Device;

class Pipeline {
public:
    Pipeline(Device &device);

    Pipeline(const Pipeline &) = delete;

    Pipeline(Pipeline &&other);

    virtual ~Pipeline();

    Pipeline &operator=(const Pipeline &) = delete;

    Pipeline &operator=(Pipeline &&) = delete;

    VkPipeline get_handle() const;

    const PipelineState &get_state() const;

protected:
    Device &device;

    VkPipeline handle = VK_NULL_HANDLE;

    PipelineState state;
};

class ComputePipeline : public Pipeline {
public:
    ComputePipeline(ComputePipeline &&) = default;

    virtual ~ComputePipeline() = default;

    ComputePipeline(Device &device,
                    VkPipelineCache pipeline_cache,
                    PipelineState &pipeline_state);
};

class GraphicsPipeline : public Pipeline {
public:
    GraphicsPipeline(GraphicsPipeline &&) = default;

    virtual ~GraphicsPipeline() = default;

    GraphicsPipeline(Device &device,
                     VkPipelineCache pipeline_cache,
                     PipelineState &pipeline_state);
};
}// namespace vox
