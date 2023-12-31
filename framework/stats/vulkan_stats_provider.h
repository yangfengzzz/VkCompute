//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/query_pool.h"
#include "stats_provider.h"

#include <utility>

namespace vox {
namespace rendering {
class RenderContext;
}// namespace rendering

class VulkanStatsProvider : public StatsProvider {
private:
    struct StatData {
        StatScaling scaling;
        uint32_t counter_index{};
        uint32_t divisor_counter_index{};
        VkPerformanceCounterStorageKHR storage;
        VkPerformanceCounterStorageKHR divisor_storage;
        StatGraphData graph_data;

        StatData() = default;

        StatData(uint32_t counter_index, VkPerformanceCounterStorageKHR storage,
                 StatScaling stat_scaling = StatScaling::ByDeltaTime,
                 uint32_t divisor_index = std::numeric_limits<uint32_t>::max(),
                 VkPerformanceCounterStorageKHR divisor_storage = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR) : scaling(stat_scaling),
                                                                                                                counter_index(counter_index),
                                                                                                                divisor_counter_index(divisor_index),
                                                                                                                storage(storage),
                                                                                                                divisor_storage(divisor_storage) {}
    };

    struct VendorStat {
        VendorStat(std::string name, const std::string &divisor_name = "")
            : name(std::move(name)),
              divisor_name(divisor_name) {
            if (!divisor_name.empty())
                scaling = StatScaling::ByCounter;
        }

        void set_vendor_graph_data(const StatGraphData &data) {
            has_vendor_graph_data = true;
            graph_data = data;
        }

        std::string name;
        StatScaling scaling = StatScaling::ByDeltaTime;
        std::string divisor_name;
        bool has_vendor_graph_data = false;
        StatGraphData graph_data;
    };

    using StatDataMap = std::unordered_map<StatIndex, StatData, StatIndexHash>;
    using VendorStatMap = std::unordered_map<StatIndex, VendorStat, StatIndexHash>;

public:
    /**
	 * @brief Constructs a VulkanStatsProvider
	 * @param requested_stats Set of stats to be collected. Supported stats will be removed from the set.
	 * @param sampling_config Sampling mode configuration (polling or continuous)
	 * @param render_context The render context
	 */
    VulkanStatsProvider(std::set<StatIndex> &requested_stats, const CounterSamplingConfig &sampling_config,
                        rendering::RenderContext &render_context);

    /**
	 * @brief Destructs a VulkanStatsProvider
	 */
    ~VulkanStatsProvider() override;

    /**
	 * @brief Checks if this provider can supply the given enabled stat
	 * @param index The stat index
	 * @return True if the stat is available, false otherwise
	 */
    [[nodiscard]] bool is_available(StatIndex index) const override;

    /**
	 * @brief Retrieve graphing data for the given enabled stat
	 * @param index The stat index
	 */
    [[nodiscard]] const StatGraphData &get_graph_data(StatIndex index) const override;

    /**
	 * @brief Retrieve a new sample set from polled sampling
	 * @param delta_time Time since last sample
	 */
    Counters sample(float delta_time) override;

    /**
	 * @brief A command buffer that we want stats about has just begun
	 * @param cb The command buffer
	 */
    void begin_sampling(core::CommandBuffer &cb) override;

    /**
	 * @brief A command buffer that we want stats about is about to be ended
	 * @param cb The command buffer
	 */
    void end_sampling(core::CommandBuffer &cb) override;

private:
    [[nodiscard]] bool is_supported(const CounterSamplingConfig &sampling_config) const;

    bool fill_vendor_data();

    bool create_query_pools(uint32_t queue_family_index);

    [[nodiscard]] float get_best_delta_time(float sw_delta_time) const;

private:
    // The render context
    rendering::RenderContext &render_context;

    // The query pool for the performance queries
    std::unique_ptr<core::QueryPool> query_pool;

    // Do we support timestamp queries
    bool has_timestamps{false};

    // The timestamp period
    float timestamp_period{1.0f};

    // Query pool for timestamps
    std::unique_ptr<core::QueryPool> timestamp_pool;

    // Map of vendor specific stat data
    VendorStatMap vendor_data;

    // Only stats which are available and were requested end up in stat_data
    StatDataMap stat_data;

    // An ordered list of the Vulkan counter ids
    std::vector<uint32_t> counter_indices;

    // How many queries have been ended?
    uint32_t queries_ready = 0;
};

}// namespace vox
