//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/query_pool.h"

namespace vox::core {
class TimestampQueryPool {
public:
    TimestampQueryPool(core::Device &device, uint32_t query_count);

    [[nodiscard]] const core::QueryPool &get_query_pool() const { return *query_pool; }
    [[nodiscard]] uint32_t get_query_count() const { return query_count; }

    [[nodiscard]] float get_nanoseconds_per_timestamp() const;

    [[nodiscard]] uint32_t get_valid_timestamp_bits() const;

    // Calculates the number of seconds elapsed between the query with index
    // |start| and |end|.
    double calculate_elapsed_seconds_between(int start, int end);

private:
    uint32_t query_count{};
    core::Device& device;
    std::unique_ptr<core::QueryPool> query_pool{nullptr};
};
}// namespace vox::core