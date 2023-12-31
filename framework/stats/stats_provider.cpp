//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "stats_provider.h"

namespace vox {
// Default graphing values for stats. May be overridden by individual providers.
std::map<StatIndex, StatGraphData> StatsProvider::default_graph_map{
    // clang-format off
    // StatIndex                        Name shown in graph                            Format           Scale                         Fixed_max Max_value
    {StatIndex::frame_times,           {"Frame Times",                                 "{:3.1f} ms",    1000.0f}},
    {StatIndex::cpu_cycles,            {"CPU Cycles",                                  "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_instructions,      {"CPU Instructions",                            "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_cache_miss_ratio,  {"Cache Miss Ratio",                            "{:3.1f}%",      100.0f,                       true,     100.0f}},
    {StatIndex::cpu_branch_miss_ratio, {"Branch Miss Ratio",                           "{:3.1f}%",      100.0f,                       true,     100.0f}},
    {StatIndex::cpu_l1_accesses,       {"CPU L1 Accesses",                             "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_instr_retired,     {"CPU Instructions Retired",                    "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_l2_accesses,       {"CPU L2 Accesses",                             "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_l3_accesses,       {"CPU L3 Accesses",                             "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_bus_reads,         {"CPU Bus Read Beats",                          "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_bus_writes,        {"CPU Bus Write Beats",                         "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_mem_reads,         {"CPU Memory Read Instructions",                "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_mem_writes,        {"CPU Memory Write Instructions",               "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_ase_spec,          {"CPU Speculatively Exec. SIMD Instructions",   "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_vfp_spec,          {"CPU Speculatively Exec. FP Instructions",     "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::cpu_crypto_spec,       {"CPU Speculatively Exec. Crypto Instructions", "{:4.1f} M/s",   static_cast<float>(1e-6)}},

    {StatIndex::gpu_cycles,            {"GPU Cycles",                                  "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_vertex_cycles,     {"Vertex Cycles",                               "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_load_store_cycles, {"Load Store Cycles",                           "{:4.0f} k/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_tiles,             {"Tiles",                                       "{:4.1f} k/s",   static_cast<float>(1e-3)}},
    {StatIndex::gpu_killed_tiles,      {"Tiles killed by CRC match",                   "{:4.1f} k/s",   static_cast<float>(1e-3)}},
    {StatIndex::gpu_fragment_jobs,     {"Fragment Jobs",                               "{:4.0f}/s"}},
    {StatIndex::gpu_fragment_cycles,   {"Fragment Cycles",                             "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_tex_cycles,        {"Shader Texture Cycles",                       "{:4.0f} k/s",   static_cast<float>(1e-3)}},
    {StatIndex::gpu_ext_reads,         {"External Reads",                              "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_ext_writes,        {"External Writes",                             "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_ext_read_stalls,   {"External Read Stalls",                        "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_ext_write_stalls,  {"External Write Stalls",                       "{:4.1f} M/s",   static_cast<float>(1e-6)}},
    {StatIndex::gpu_ext_read_bytes,    {"External Read Bytes",                         "{:4.1f} MiB/s", 1.0f / (1024.0f * 1024.0f)}},
    {StatIndex::gpu_ext_write_bytes,   {"External Write Bytes",                        "{:4.1f} MiB/s", 1.0f / (1024.0f * 1024.0f)}},
    // clang-format on
};

// Static
const StatGraphData &StatsProvider::default_graph_data(StatIndex index) {
    return default_graph_map.at(index);
}

}// namespace vox
