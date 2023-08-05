//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "platform/platform.h"

namespace vox {
enum UnixType {
    Mac,
    Linux
};

class UnixPlatform : public Platform {
public:
    UnixPlatform(const PlatformContext &context, const UnixType &type);

    virtual ~UnixPlatform() = default;

protected:
    virtual void create_window(const Window::Properties &properties) override;

private:
    UnixType type;
};
}// namespace vox
