//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unix_d2d_platform.h"

#include "common/error.h"

VKBP_DISABLE_WARNINGS()
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
VKBP_ENABLE_WARNINGS()

#include "platform/headless_window.h"
#include "platform/unix/direct_window.h"

namespace vox {
namespace {
inline const std::string get_temp_path_from_environment() {
    std::string temp_path = "/tmp/";

    if (const char *env_ptr = std::getenv("TMPDIR")) {
        temp_path = std::string(env_ptr) + "/";
    }

    return temp_path;
}
}// namespace

namespace fs {
void create_directory(const std::string &path) {
    if (!is_directory(path)) {
        mkdir(path.c_str(), 0777);
    }
}
}// namespace fs

UnixD2DPlatform::UnixD2DPlatform() : Platform{} {
}

void UnixD2DPlatform::create_window(const Window::Properties &properties) {
    if (properties.mode == vox::Window::Mode::Headless) {
        window = std::make_unique<HeadlessWindow>(properties);
    } else {
        window = std::make_unique<DirectWindow>(this, properties);
    }
}
}// namespace vox
