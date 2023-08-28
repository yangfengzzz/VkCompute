//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "framework/common/logging.h"
#include "framework/platform/platform.h"
#include "framework/platform/core/entrypoint.hpp"

#if defined(PLATFORM__ANDROID)
#include "platform/android/android_platform.h"
#elif defined(PLATFORM__WINDOWS)
#include "platform/windows/windows_platform.h"
#elif defined(PLATFORM__LINUX_D2D)
#include "platform/unix/unix_d2d_platform.h"
#elif defined(PLATFORM__LINUX) || defined(PLATFORM__MACOS)
#include "platform/unix/unix_platform.h"
#else
#error "Platform not supported"
#endif

#include "primitive_app.h"
#include "atomic_compute_app.h"
#include "cuda_compute_app.h"

CUSTOM_MAIN(context) {
#if defined(PLATFORM__ANDROID)
    vox::AndroidPlatform platform{context};
#elif defined(PLATFORM__WINDOWS)
    vox::WindowsPlatform platform{context};
#elif defined(PLATFORM__LINUX_D2D)
    vox::UnixD2DPlatform platform{context};
#elif defined(PLATFORM__LINUX)
    vox::UnixPlatform platform{context, vox::UnixType::Linux};
#elif defined(PLATFORM__MACOS)
    vox::UnixPlatform platform{context, vox::UnixType::Mac};
#else
#error "Platform not supported"
#endif

    auto code = platform.initialize();

    if (code == vox::ExitCode::Success) {
        auto app = std::make_unique<vox::CudaComputeApp>();
        app->prepare({false, &platform.get_window()});
        platform.set_callback([&](float delta_time) { app->update(delta_time); },
                              [&](uint32_t width, uint32_t height) {
                                  app->resize(width, height, 1, 1);
                              },
                              [&](const vox::InputEvent &event) {
                                  app->input_event(event);
                              });

        // loop
        code = platform.main_loop();
    }

    platform.terminate(code);

    return 0;
}