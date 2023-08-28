//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_compute_app.h"

namespace vox {
bool CudaComputeApp::prepare(const ApplicationOptions &options) {
    ForwardApplication::prepare(options);
    cuda_device = std::make_unique<compute::CudaDevice>(device->get_gpu().get_device_id_properties().deviceUUID, VK_UUID_SIZE);
    cuda_stream = std::make_unique<compute::CudaStream>(*cuda_device);
    cuda_sim = std::make_unique<compute::SineWaveSimulation>(0, 0, *cuda_device);
    return true;
}

void CudaComputeApp::load_scene() {}

void CudaComputeApp::update_gpu_task(core::CommandBuffer &command_buffer) {
    cuda_sim->step_simulation(0, nullptr, *cuda_stream);
}

}// namespace vox