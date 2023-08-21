//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>
#include <vector>

#include <core/platform/context.hpp>

#include <game-activity/native_app_glue/android_native_app_glue.h>

namespace vox {
/**
 * @brief Android platform context
 *
 * @warning Use in extreme circumstances with code guarded by the PLATFORM__ANDROID define
 */
class AndroidPlatformContext final : public PlatformContext {
public:
    AndroidPlatformContext(android_app *app);
    ~AndroidPlatformContext() override = default;

    android_app *app{nullptr};

    static std::string android_external_storage_directory;
    static std::string android_temp_directory;
    static std::vector<std::string> android_arguments;
};
}// namespace vox