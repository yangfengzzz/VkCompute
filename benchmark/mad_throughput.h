//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "benchmark_api.h"
#include "compute/data_type_util.h"
#include "compute/compute_pass.h"

namespace vox::benchmark {
class MADThroughPut : public BenchmarkAPI {
public:
    void register_vulkan_benchmarks(compute::ComputeResource &resource) override;

private:
    std::unique_ptr<compute::ComputePass> pass{nullptr};
    std::unique_ptr<ShaderData> shader_data;
};

}// namespace vox