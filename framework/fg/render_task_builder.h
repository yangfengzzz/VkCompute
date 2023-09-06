//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/device.h"
#include <thsvs_simpler_vulkan_synchronization.h>
#include "fg/common.h"

namespace vox::fg {
class Framegraph;
class RenderTaskBase;

// The interface between the framegraph and a render task.
class RenderTaskBuilder {
public:
    explicit RenderTaskBuilder(Framegraph *framegraph, RenderTaskBase *render_task)
        : framegraph_(framegraph), render_task_(render_task) {
    }
    RenderTaskBuilder(const RenderTaskBuilder &that) = default;
    RenderTaskBuilder(RenderTaskBuilder &&temp) = default;
    virtual ~RenderTaskBuilder() = default;
    RenderTaskBuilder &operator=(const RenderTaskBuilder &that) = default;
    RenderTaskBuilder &operator=(RenderTaskBuilder &&temp) = default;

    template<typename resource_type, typename description_type>
    resource_type *create(const std::string &name, const description_type &description);

    template<typename resource_type>
    resource_type *read(resource_type *resource, ThsvsAccessType access_type,
                        PassResourceAccessSyncType sync_type = PassResourceAccessSyncType::AlwaysSync);

    template<typename resource_type>
    resource_type *write(resource_type *resource, ThsvsAccessType access_type,
                         PassResourceAccessSyncType sync_type = PassResourceAccessSyncType::AlwaysSync);

private:
    Framegraph *framegraph_;
    RenderTaskBase *render_task_;
};
}// namespace vox::fg
