//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <core/platform/entrypoint.hpp>

#include "windows/context.hpp"
#include <Windows.h>

std::unique_ptr<vox::PlatformContext> create_platform_context(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow) {
    return std::make_unique<vox::WindowsPlatformContext>(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
}
