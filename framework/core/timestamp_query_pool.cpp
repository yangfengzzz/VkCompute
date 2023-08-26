//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/timestamp_query_pool.h"
#include "core/device.h"

namespace vox::core {
TimestampQueryPool::TimestampQueryPool(core::Device &device, uint32_t query_count)
    : query_count{query_count},
      device{device} {
    if (get_valid_timestamp_bits() == 0) {
        LOGE("the device does not support timestamp");
    }

    // Create query pool.
    VkQueryPoolCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    create_info.pNext = nullptr;
    create_info.flags = 0;
    create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create_info.queryCount = query_count;
    create_info.pipelineStatistics = 0;
    query_pool = std::make_unique<QueryPool>(device, create_info);
}

float TimestampQueryPool::get_nanoseconds_per_timestamp() const {
    return device.get_gpu().get_properties().limits.timestampPeriod;
}

uint32_t TimestampQueryPool::get_valid_timestamp_bits() const {
    return device.get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0).get_properties().timestampValidBits;
}

double TimestampQueryPool::calculate_elapsed_seconds_between(int start, int end) {
    uint32_t count = end - start + 1;
    std::vector<uint64_t> timestamps(count);
    query_pool->get_results(start, count, count * sizeof(uint64_t),
                            reinterpret_cast<void *>(timestamps.data()), sizeof(uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return double(timestamps.back() - timestamps.front()) *
           get_nanoseconds_per_timestamp() * 1e-9;
}

}// namespace vox::core