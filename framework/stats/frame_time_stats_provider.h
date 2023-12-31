//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "stats/stats_provider.h"

namespace vox {
class FrameTimeStatsProvider : public StatsProvider {
public:
    /**
	 * @brief Constructs a FrameTimeStatsProvider
	 * @param requested_stats Set of stats to be collected. Supported stats will be removed from the set.
	 */
    FrameTimeStatsProvider(std::set<StatIndex> &requested_stats);
    /**
	 * @brief Checks if this provider can supply the given enabled stat
	 * @param index The stat index
	 * @return True if the stat is available, false otherwise
	 */
    [[nodiscard]] bool is_available(StatIndex index) const override;

    /**
	 * @brief Retrieve a new sample set
	 * @param delta_time Time since last sample
	 */
    Counters sample(float delta_time) override;
};
}// namespace vox
