//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <core/platform/entrypoint.hpp>

#include "android/context.hpp"

std::unique_ptr<vox::PlatformContext> create_platform_context(android_app *app) {
    return std::make_unique<vox::AndroidPlatformContext>(app);
}