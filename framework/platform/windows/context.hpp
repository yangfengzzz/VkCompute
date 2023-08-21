//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

#include <Windows.h>
#include <core/platform/context.hpp>

namespace vox {
/**
 * @brief MS Windows platform context
 *
 * @warning Use in extreme circumstances with code guarded by the PLATFORM__WINDOWS define
 */
class WindowsPlatformContext final : public PlatformContext {
public:
    WindowsPlatformContext(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow);
    ~WindowsPlatformContext() override = default;
};
}// namespace vox