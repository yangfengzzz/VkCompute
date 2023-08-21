/* Copyright (c) 2019-2023, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

    auto app = std::make_unique<vox::AtomicComputeApp>();
    app->prepare({false, &platform.get_window()});
    auto code = platform.initialize(
        [&](float delta_time) {
            app->update(delta_time);
        },
        [&](uint32_t width, uint32_t height) {
            app->resize(width, height, 1, 1);
        },
        [&](const vox::InputEvent &event) {
            app->input_event(event);
        });

    if (code == vox::ExitCode::Success) {
        code = platform.main_loop();
    }

    platform.terminate(code);

    return 0;
}