//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <benchmark/benchmark.h>
#include "compute/compute_context.h"
#include "mad_throughput.h"

using namespace vox::compute;

int main(int argc, char **argv) {
    ::benchmark::Initialize(&argc, argv);
    auto context = ComputeContext();

    size_t device_index = 1;
    auto app = std::make_unique<vox::benchmark::MADThroughPut>();
    auto resource = context.get_resource(device_index);
    app->register_vulkan_benchmarks(resource);

    ::benchmark::RunSpecifiedBenchmarks();
}