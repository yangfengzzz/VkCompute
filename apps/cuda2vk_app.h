//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "application/forward_application.h"
#include "mesh/buffer_mesh.h"
#include "cuda/core/cuda_device.h"
#include "cuda/core/cuda_stream.h"
#include "cuda/core/cuda_external_buffer.h"
#include "cuda/core/cuda_external_semaphore.h"
#include "cuda/solver/monte_carlo_pi.h"

namespace vox {
class Cuda2VkApp : public ForwardApplication {
public:
    static constexpr size_t NUM_SIMULATION_POINTS = 50000;

    void before_prepare() override;

    void after_prepare() override;

    Camera *load_scene() override;

public:
    static void get_vertex_descriptions(
        std::vector<VkVertexInputBindingDescription> &bindingDesc,
        std::vector<VkVertexInputAttributeDescription> &attribDesc);

    compute::CudaDevice &get_cuda_device() {
        return *cuda_device;
    }

    std::shared_ptr<Material> get_material() {
        return material_;
    }

    compute::MonteCarloPiSimulation &get_cuda_sim() {
        return *cuda_sim;
    }

    core::Semaphore &get_wait_semaphore() {
        return *wait_semaphore;
    }

    core::Semaphore &get_signal_semaphore() {
        return *signal_semaphore;
    }

private:
    std::unique_ptr<compute::CudaDevice> cuda_device{nullptr};
    std::unique_ptr<compute::MonteCarloPiSimulation> cuda_sim{nullptr};

    std::unique_ptr<core::Buffer> xy_position_buffer{nullptr};
    std::unique_ptr<core::Buffer> in_circle_buffer{nullptr};

    std::shared_ptr<BufferMesh> mesh{nullptr};
    std::shared_ptr<Material> material_{nullptr};

    std::unique_ptr<core::Semaphore> wait_semaphore{nullptr};
    std::unique_ptr<core::Semaphore> signal_semaphore{nullptr};
};

}// namespace vox
