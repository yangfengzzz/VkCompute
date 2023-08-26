//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mad_throughput.h"
#include "shader/shader_module.h"
#include <spdlog/fmt/fmt.h>

namespace vox {
static void throughput(::benchmark::State &state,
                       compute::ComputeResource *resource,
                       compute::ComputePass *pass,
                       ShaderData *shader_data,
                       size_t num_element, int loop_count, DataType data_type) {
    core::CommandBuffer &cmd = resource->begin();
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    pass->set_dispatch_size({(uint32_t)num_element / (4 * 16), 1, 1});
    pass->compute(cmd);
    cmd.end();
    resource->submit(cmd);
}

void MADThroughPut::register_vulkan_benchmarks(compute::ComputeResource &resource) {
    const char *gpu_name = resource.gpu.get_properties().deviceName;

    ShaderVariant shader_variant;
    auto shader = std::make_shared<ShaderModule>(resource.get_device(), VK_SHADER_STAGE_COMPUTE_BIT,
                                                 "compute/mad_throughput.glsl", "main", shader_variant);
    pass = std::make_unique<compute::ComputePass>(shader);
    shader_data = std::make_unique<ShaderData>(resource.get_device());
    pass->attach_shader_data(shader_data.get());

    const size_t num_element = 1024 * 1024;
    const int min_loop_count = 100000;
    const int max_loop_count = min_loop_count * 2;

    for (int loop_count = min_loop_count; loop_count <= max_loop_count;
         loop_count += min_loop_count) {
        std::string test_name = fmt::format("{}/{}/{}/{}", gpu_name, "mad_throughput", num_element, loop_count);
        ::benchmark::RegisterBenchmark(test_name, throughput, &resource, pass.get(), shader_data.get(),
                                       num_element, loop_count, DataType::fp32)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
    }
}

}// namespace vox