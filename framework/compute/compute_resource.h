//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/frame_resource.h"

namespace vox::compute {
enum class LatencyMeasureMode {
    // time spent from queue submit to returning from queue wait
    kSystemSubmit,
    // system_submit subtracted by time for void dispatch
    kSystemDispatch,
    // Timestamp difference measured on GPU
    kGpuTimestamp,
};

struct LatencyMeasure {
    LatencyMeasureMode mode = LatencyMeasureMode::kSystemSubmit;
    double overhead_seconds{};
};

class ComputeResource : public core::FrameResource {
public:
    LatencyMeasure latency_measure{};

    ComputeResource(core::PhysicalDevice &gpu, core::Device &device, size_t thread_count = 1);

    ComputeResource(const ComputeResource &) = delete;

    ComputeResource(ComputeResource &&) = delete;

    ComputeResource &operator=(const ComputeResource &) = delete;

    ComputeResource &operator=(ComputeResource &&) = delete;

public:
    core::PhysicalDevice &gpu;

    /**
	 * @brief Prepares the next available frame for rendering
	 * @param reset_mode How to reset the command buffer
	 * @returns A valid command buffer to record commands to be submitted
	 * Also ensures that there is an active frame if there is no existing active frame already
	 */
    core::CommandBuffer &begin(core::CommandBuffer::ResetMode reset_mode = core::CommandBuffer::ResetMode::ResetPool);

    /**
	 * @brief Submits the command buffer to the right queue
	 * @param command_buffer A command buffer containing recorded commands
	 */
    void submit(core::CommandBuffer &command_buffer);

    /**
	 * @brief Submits multiple command buffers to the right queue
	 * @param command_buffers Command buffers containing recorded commands
	 */
    void submit(const std::vector<core::CommandBuffer *> &command_buffers);

    /**
	 * @brief Submits a command buffer related to a frame to a queue
	 */
    void submit(const core::Queue &queue, const std::vector<core::CommandBuffer *> &command_buffers);
};

}// namespace vox::compute
