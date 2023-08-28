//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "application/forward_application.h"
#include "mesh/buffer_mesh.h"
#include "cuda/cuda_device.h"
#include "cuda/cuda_stream.h"
#include "cuda/cuda_external_buffer.h"
#include "cuda/cuda_external_semaphore.h"
#include "cuda/sine_wave_simulation.h"

namespace vox {
class CudaComputeApp : public ForwardApplication {
public:
    bool prepare(const ApplicationOptions &options) override;

    void load_scene() override;

public:
    void init_buffer(core::Buffer &index_buffer);

    static void get_vertex_descriptions(
        std::vector<VkVertexInputBindingDescription> &bindingDesc,
        std::vector<VkVertexInputAttributeDescription> &attribDesc);

    compute::CudaDevice &get_cuda_device() {
        return *cuda_device;
    }

    compute::SineWaveSimulation &get_cuda_sim() {
        return *cuda_sim;
    }

    core::Buffer &get_height_buffer() {
        return *height_buffer;
    }

    core::Semaphore &get_wait_semaphore() {
        return *wait_semaphore;
    }

    core::Semaphore &get_signal_semaphore() {
        return *signal_semaphore;
    }

private:
    std::unique_ptr<compute::CudaDevice> cuda_device{nullptr};
    std::unique_ptr<compute::SineWaveSimulation> cuda_sim{nullptr};

    std::unique_ptr<core::BufferPool> external_pool{nullptr};
    std::unique_ptr<core::Buffer> height_buffer{nullptr};
    std::unique_ptr<core::Buffer> xy_buffer{nullptr};

    std::shared_ptr<BufferMesh> mesh{nullptr};
    std::shared_ptr<Material> material_{nullptr};

    std::unique_ptr<core::Semaphore> wait_semaphore{nullptr};
    std::unique_ptr<core::Semaphore> signal_semaphore{nullptr};
};

}// namespace vox
