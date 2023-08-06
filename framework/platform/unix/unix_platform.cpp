//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unix_platform.h"

#include "common/error.h"

#include "platform/glfw_window.h"
#include "platform/headless_window.h"

VKBP_DISABLE_WARNINGS()
#include <spdlog/sinks/stdout_color_sinks.h>
VKBP_ENABLE_WARNINGS()

#ifndef VK_MVK_MACOS_SURFACE_EXTENSION_NAME
#define VK_MVK_MACOS_SURFACE_EXTENSION_NAME "VK_MVK_macos_surface"
#endif

#ifndef VK_KHR_XCB_SURFACE_EXTENSION_NAME
#define VK_KHR_XCB_SURFACE_EXTENSION_NAME "VK_KHR_xcb_surface"
#endif

#ifndef VK_EXT_METAL_SURFACE_EXTENSION_NAME
#define VK_EXT_METAL_SURFACE_EXTENSION_NAME "VK_EXT_metal_surface"
#endif

#ifndef VK_KHR_XLIB_SURFACE_EXTENSION_NAME
#define VK_KHR_XLIB_SURFACE_EXTENSION_NAME "VK_KHR_xlib_surface"
#endif

#ifndef VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME
#define VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME "VK_KHR_wayland_surface"
#endif

namespace vox {

namespace fs {
void create_directory(const std::string &path) {
    if (!is_directory(path)) {
        mkdir(path.c_str(), 0777);
    }
}
}// namespace fs

UnixPlatform::UnixPlatform(const UnixType &type) : Platform{}, type{type} {
}

void UnixPlatform::create_window(const Window::Properties &properties) {
    if (properties.mode == vox::Window::Mode::Headless) {
        window = std::make_unique<HeadlessWindow>(properties);
    } else {
        window = std::make_unique<GlfwWindow>(this, properties);
    }
}
}// namespace vox
