//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "framework/core/device.h"
#include "framework/core/pipeline_state.h"
#include "framework/shader/shader_data.h"
#include "material/enums/render_queue_type.h"

namespace vox {
/**
 * Material.
 */
class Material {
public:
    /** Name. */
    std::string name_;

    /** Render queue type. */
    RenderQueueType::Enum render_queue_ = RenderQueueType::Enum::OPAQUE;

    /** Shader used by the material. */
    std::shared_ptr<ShaderModule> vertex_source_{nullptr};
    std::shared_ptr<ShaderModule> fragment_source_{nullptr};

    /** Shader data. */
    ShaderData shader_data_;

    /** Render state. */
    core::InputAssemblyState input_assembly_state_;
    core::RasterizationState rasterization_state_;
    core::MultisampleState multisample_state_;
    core::DepthStencilState depth_stencil_state_;
    core::ColorBlendState color_blend_state_;

    explicit Material(core::Device &device, std::string name = "");

    Material(Material &&other) = default;

    virtual ~Material() = default;

protected:
    core::Device &device_;
};

}// namespace vox
