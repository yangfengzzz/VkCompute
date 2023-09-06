//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/device.h"
#include <thsvs_simpler_vulkan_synchronization.h>

namespace vox::fg {
class Framegraph;
class RenderTaskBase;
class RenderTaskBuilder;

class ResourceBase {
public:
    explicit ResourceBase(std::string name, const RenderTaskBase *creator, ThsvsAccessType access_type)
        : name_(std::move(name)), creator_(creator), ref_count_(0), access_type_(access_type) {
        static std::size_t id = 0;
        id_ = id++;
    }
    ResourceBase(const ResourceBase &that) = delete;
    ResourceBase(ResourceBase &&temp) = default;
    virtual ~ResourceBase() = default;
    ResourceBase &operator=(const ResourceBase &that) = delete;
    ResourceBase &operator=(ResourceBase &&temp) = default;

    [[nodiscard]] std::size_t id() const {
        return id_;
    }

    [[nodiscard]] const std::string &name() const {
        return name_;
    }

    void set_name(const std::string &name) {
        name_ = name;
    }

    [[nodiscard]] ThsvsAccessType access_type() const {
        return access_type_;
    }

    [[nodiscard]] bool transient() const {
        return creator_ != nullptr;
    }

protected:
    friend Framegraph;
    friend RenderTaskBuilder;

    virtual void realize() = 0;
    virtual void derealize() = 0;

    ThsvsAccessType access_type_;
    std::size_t id_;
    std::string name_;
    const RenderTaskBase *creator_;
    std::vector<const RenderTaskBase *> readers_;
    std::vector<const RenderTaskBase *> writers_;
    std::size_t ref_count_;// Computed through framegraph compilation.
};

}// namespace vox::fg