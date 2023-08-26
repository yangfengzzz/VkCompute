//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mad_throughput.h"
#include "compute/buffer_utils.h"
#include "compute/status_util.h"
#include <spdlog/fmt/fmt.h>
#include <gtest/gtest.h>

namespace vox::benchmark {
static void throughput(::benchmark::State &state,
                       compute::ComputeResource *resource,
                       compute::ComputePass *pass,
                       ShaderData *shader_data,
                       size_t num_element, int loop_count, compute::DataType data_type) {
    auto &device = resource->get_device();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    const size_t src0_size = num_element * get_size(data_type);
    const size_t src1_size = num_element * get_size(data_type);
    const size_t dst_size = num_element * get_size(data_type);

    auto src0_buffer = core::Buffer(device, src0_size,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_GPU_ONLY);
    auto src1_buffer = core::Buffer(device, src1_size,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_GPU_ONLY);
    auto dst_buffer = core::Buffer(device, dst_size,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VMA_MEMORY_USAGE_GPU_ONLY);

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    auto getSrc0 = [](size_t i) {
        float v = float((i % 9) + 1) * 0.1f;
        return v;
    };
    auto getSrc1 = [](size_t i) {
        float v = float((i % 5) + 1) * 1.f;
        return v;
    };

    if (data_type == compute::DataType::fp16) {
        compute::set_device_buffer_via_staging_buffer(
            device, src0_buffer, src0_size, [&](void *ptr, size_t num_bytes) {
                auto *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
                for (size_t i = 0; i < num_element; i++) {
                    src_float_buffer[i] = compute::fp16(getSrc0(i)).get_value();
                }
            });

        compute::set_device_buffer_via_staging_buffer(
            device, src1_buffer, src1_size, [&](void *ptr, size_t num_bytes) {
                auto *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
                for (size_t i = 0; i < num_element; i++) {
                    src_float_buffer[i] = compute::fp16(getSrc1(i)).get_value();
                }
            });
    } else if (data_type == compute::DataType::fp32) {
        compute::set_device_buffer_via_staging_buffer(
            device, src0_buffer, src0_size, [&](void *ptr, size_t num_bytes) {
                auto *src_float_buffer = reinterpret_cast<float *>(ptr);
                for (size_t i = 0; i < num_element; i++) {
                    src_float_buffer[i] = getSrc0(i);
                }
            });

        compute::set_device_buffer_via_staging_buffer(
            device, src1_buffer, src1_size, [&](void *ptr, size_t num_bytes) {
                auto *src_float_buffer = reinterpret_cast<float *>(ptr);
                for (size_t i = 0; i < num_element; i++) {
                    src_float_buffer[i] = getSrc1(i);
                }
            });
    }

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/
    shader_data->set_buffer_functor("inputA", [&]() -> core::Buffer * { return &src0_buffer; });
    shader_data->set_buffer_functor("inputB", [&]() -> core::Buffer * { return &src1_buffer; });
    shader_data->set_buffer_functor("Output", [&]() -> core::Buffer * { return &dst_buffer; });

    core::CommandBuffer &cmd = resource->begin();
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmd.set_specialization_constant(0, loop_count);
    pass->set_dispatch_size({(uint32_t)num_element / (4 * 16), 1, 1});
    pass->compute(cmd);
    cmd.end();
    resource->submit(cmd);
    device.get_fence_pool().wait();

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/

    if (data_type == compute::DataType::fp16) {
        compute::get_device_buffer_via_staging_buffer(
            device, dst_buffer, dst_size, [&](void *ptr, size_t num_bytes) {
                auto *dst_float_buffer = reinterpret_cast<uint16_t *>(ptr);
                for (size_t i = 0; i < num_element; i++) {
                    float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
                    BM_CHECK_FLOAT_EQ(compute::fp16(dst_float_buffer[i]).to_float(), limit, 0.5f)
                        << "destination buffer element #" << i
                        << " has incorrect value: expected to be " << limit
                        << " but found " << compute::fp16(dst_float_buffer[i]).to_float();
                }
            });
    } else if (data_type == compute::DataType::fp32) {
        compute::get_device_buffer_via_staging_buffer(
            device, dst_buffer, dst_size, [&](void *ptr, size_t num_bytes) {
                auto *dst_float_buffer = reinterpret_cast<float *>(ptr);
                for (size_t i = 0; i < num_element; i++) {
                    float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
                    BM_CHECK_FLOAT_EQ(dst_float_buffer[i], limit, 0.01f)
                        << "destination buffer element #" << i
                        << " has incorrect value: expected to be " << limit
                        << " but found " << dst_float_buffer[i];
                }
            });
    }
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
                                       num_element, loop_count, compute::DataType::fp32)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
    }
}

}// namespace vox::benchmark