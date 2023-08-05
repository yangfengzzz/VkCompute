//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "query_pool.h"

#include "device.h"

namespace vox {
QueryPool::QueryPool(Device &d, const VkQueryPoolCreateInfo &info) : device{d} {
    VK_CHECK(vkCreateQueryPool(device.get_handle(), &info, nullptr, &handle));
}

QueryPool::QueryPool(QueryPool &&other) : device{other.device},
                                          handle{other.handle} {
    other.handle = VK_NULL_HANDLE;
}

QueryPool::~QueryPool() {
    if (handle != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device.get_handle(), handle, nullptr);
    }
}

VkQueryPool QueryPool::get_handle() const {
    assert(handle != VK_NULL_HANDLE && "QueryPool handle is invalid");
    return handle;
}

void QueryPool::host_reset(uint32_t first_query, uint32_t query_count) {
    assert(device.is_enabled("VK_EXT_host_query_reset") &&
           "VK_EXT_host_query_reset needs to be enabled to call QueryPool::host_reset");

    vkResetQueryPoolEXT(device.get_handle(), get_handle(), first_query, query_count);
}

VkResult QueryPool::get_results(uint32_t first_query, uint32_t num_queries,
                                size_t result_bytes, void *results, VkDeviceSize stride,
                                VkQueryResultFlags flags) {
    return vkGetQueryPoolResults(device.get_handle(), get_handle(), first_query, num_queries,
                                 result_bytes, results, stride, flags);
}

}// namespace vox
