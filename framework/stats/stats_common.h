//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <chrono>
#include <string>

#if defined(VK_USE_PLATFORM_XLIB_KHR)
#undef None
#endif

namespace vox {
/**
 * @brief Handles of stats to be optionally enabled in @ref vox::Stats
 */
enum class StatIndex {
    frame_times,
    cpu_cycles,
    cpu_instructions,
    cpu_cache_miss_ratio,
    cpu_branch_miss_ratio,
    cpu_l1_accesses,
    cpu_instr_retired,
    cpu_l2_accesses,
    cpu_l3_accesses,
    cpu_bus_reads,
    cpu_bus_writes,
    cpu_mem_reads,
    cpu_mem_writes,
    cpu_ase_spec,
    cpu_vfp_spec,
    cpu_crypto_spec,

    gpu_cycles,
    gpu_vertex_cycles,
    gpu_load_store_cycles,
    gpu_tiles,
    gpu_killed_tiles,
    gpu_fragment_jobs,
    gpu_fragment_cycles,
    gpu_ext_reads,
    gpu_ext_writes,
    gpu_ext_read_stalls,
    gpu_ext_write_stalls,
    gpu_ext_read_bytes,
    gpu_ext_write_bytes,
    gpu_tex_cycles,
};

struct StatIndexHash {
    template<typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

enum class StatScaling {
    // The stat is not scaled
    None,

    // The stat is scaled by delta time, useful for per-second values
    ByDeltaTime,

    // The stat is scaled by another counter, useful for ratios
    ByCounter
};

enum class CounterSamplingMode {
    /// Sample counters only when calling update()
    Polling,
    /// Sample counters continuously, update circular buffers when calling update()
    Continuous
};

struct CounterSamplingConfig {
    /// Sampling mode (polling or continuous)
    CounterSamplingMode mode;

    /// Sampling interval in continuous mode
    std::chrono::milliseconds interval{1};

    /// Speed of circular buffer updates in continuous mode;
    /// at speed = 1.0f a new sample is displayed over 1 second.
    float speed{0.5f};
};

// Per-statistic graph data
class StatGraphData {
public:
    /**
	 * @brief Constructs data for the graph
	 * @param name Name of the Stat
	 * @param format Format of the label
	 * @param scale_factor Any scaling to apply to the data
	 * @param has_fixed_max Whether the data should have a fixed max value
	 * @param max_value The maximum value to use
	 */
    StatGraphData(std::string name,
                  std::string format,
                  float scale_factor = 1.0f,
                  bool has_fixed_max = false,
                  float max_value = 0.0f);

    StatGraphData() = default;

    std::string name;
    std::string format;
    float scale_factor{};
    bool has_fixed_max{};
    float max_value{};
};

}// namespace vox
