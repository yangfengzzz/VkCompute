//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>
#include <string>
#include <variant>

#include "fg/realize.h"
#include "fg/resource_base.h"

namespace vox::fg {
class RenderTaskBase;

template<typename description_type_, typename actual_type_>
class Resource final : public ResourceBase {
public:
    using description_type = description_type_;
    using actual_type = actual_type_;

    explicit Resource(const std::string &name, const RenderTaskBase *creator, const description_type &description)
        : ResourceBase(name, creator, THSVS_ACCESS_NONE), description_(description), actual_(std::unique_ptr<actual_type>()) {
        // Transient (normal) constructor.
    }
    explicit Resource(const std::string &name, const description_type &description,
                      ThsvsAccessType access_type, actual_type *actual)
        : ResourceBase(name, nullptr, access_type), description_(description), actual_(actual) {
        assert(actual != nullptr);
    }
    Resource(const Resource &that) = delete;
    Resource(Resource &&temp) = default;
    ~Resource() override = default;
    Resource &operator=(const Resource &that) = delete;
    Resource &operator=(Resource &&temp) = default;

    const description_type &description() const {
        return description_;
    }

    // If transient, only valid through the realized interval of the resource.
    actual_type *actual() const {
        return std::holds_alternative<std::unique_ptr<actual_type>>(actual_) ?
                   std::get<std::unique_ptr<actual_type>>(actual_).get() :
                   std::get<actual_type *>(actual_);
    }

private:
    void realize(TransientResourceCache &cache) override {
        if (transient()) {
            std::get<std::unique_ptr<actual_type>>(actual_) = fg::realize<description_type, actual_type>(cache, description_);
        }
    }

    void derealize(TransientResourceCache &cache) override {
        if (transient()) {
            fg::derealize(cache, std::move(std::get<std::unique_ptr<actual_type>>(actual_)));
        }
    }

    description_type description_;
    std::variant<std::unique_ptr<actual_type>, actual_type *> actual_;
};
}// namespace vox::fg