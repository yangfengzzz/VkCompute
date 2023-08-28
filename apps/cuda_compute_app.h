//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "application/forward_application.h"
#include "cuda/cuda_device.h"
#include "cuda/cuda_stream.h"
#include "cuda/sine_wave_simulation.h"

namespace vox {
class CudaComputeApp : public ForwardApplication {
public:
    bool prepare(const ApplicationOptions &options) override;

    void load_scene() override;

    void update_gpu_task(core::CommandBuffer &command_buffer) override;

private:
    std::unique_ptr<compute::CudaDevice> cuda_device{nullptr};
    std::unique_ptr<compute::CudaStream> cuda_stream{nullptr};
    std::unique_ptr<compute::SineWaveSimulation> cuda_sim{nullptr};

    std::shared_ptr<Material> material_{nullptr};
};

}// namespace vox
