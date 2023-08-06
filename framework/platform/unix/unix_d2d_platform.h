//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "platform/platform.h"

namespace vox {
class UnixD2DPlatform : public Platform {
public:
    UnixD2DPlatform();

    virtual ~UnixD2DPlatform() = default;

protected:
    virtual void create_window(const Window::Properties &properties) override;
};
}// namespace vox
