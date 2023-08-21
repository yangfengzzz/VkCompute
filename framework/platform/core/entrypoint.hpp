//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "platform/core/context.hpp"

// Platform specific entrypoint definitions
// Applications should use CUSTOM_MAIN to define their own main function
// Definitions added by core/CMakeLists.txt

#if defined(PLATFORM__ANDROID)
#include <game-activity/native_app_glue/android_native_app_glue.h>
extern std::unique_ptr<vox::PlatformContext> create_platform_context(android_app *state);

#define CUSTOM_MAIN(context_name)                      \
    int platform_main(const vox::PlatformContext &);   \
    void android_main(android_app *state) {            \
        auto context = create_platform_context(state); \
        platform_main(*context);                       \
    }                                                  \
    int platform_main(const vox::PlatformContext &context_name)
#elif defined(PLATFORM__WINDOWS)
#include <Windows.h>
extern std::unique_ptr<vox::PlatformContext> create_platform_context(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow);

#define CUSTOM_MAIN(context_name)                                                                      \
    int platform_main(const vox::PlatformContext &);                                                   \
    int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow) { \
        auto context = create_platform_context(hInstance, hPrevInstance, lpCmdLine, nCmdShow);         \
        return platform_main(*context);                                                                \
    }                                                                                                  \
    int platform_main(const vox::PlatformContext &context_name)
#elif defined(PLATFORM__LINUX) || defined(PLATFORM__MACOS)
extern std::unique_ptr<vox::PlatformContext> create_platform_context(int argc, char **argv);

#define CUSTOM_MAIN(context_name)                           \
    int platform_main(const vox::PlatformContext &);        \
    int main(int argc, char *argv[]) {                      \
        auto context = create_platform_context(argc, argv); \
        return platform_main(*context);                     \
    }                                                       \
    int platform_main(const vox::PlatformContext &context_name)

#else
#include <stdexcept>
#define CUSTOM_MAIN(context_name)                           \
    int main(int argc, char *argv[]) {                      \
        throw std::runtime_error{"platform not supported"}; \
    }                                                       \
    int unused(const vox::PlatformContext &context_name)
#endif
