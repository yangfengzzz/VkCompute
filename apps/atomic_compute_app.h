//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "application/forward_application.h"
#include "framework/compute//compute_pass.h"

namespace vox {
class AtomicComputeApp : public ForwardApplication {
public:
    void after_load_scene() override;

    Camera *load_scene() override;

    void update_gpu_task(core::CommandBuffer &command_buffer) override;

private:
    std::shared_ptr<Material> material_{nullptr};
    std::unique_ptr<compute::ComputePass> atomic_pass{nullptr};
};

}// namespace vox
