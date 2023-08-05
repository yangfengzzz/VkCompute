//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "window.h"

#include "platform/platform.h"

namespace vox {
Window::Window(const Properties &properties) : properties{properties} {
}

void Window::process_events() {
}

Window::Extent Window::resize(const Extent &new_extent) {
    if (properties.resizable) {
        properties.extent.width = new_extent.width;
        properties.extent.height = new_extent.height;
    }

    return properties.extent;
}

const Window::Extent &Window::get_extent() const {
    return properties.extent;
}

float Window::get_content_scale_factor() const {
    return 1.0f;
}

Window::Mode Window::get_window_mode() const {
    return properties.mode;
}

bool Window::get_display_present_info(VkDisplayPresentInfoKHR *info,
                                      uint32_t src_width, uint32_t src_height) const {
    // Default is to not use the extra present info
    return false;
}
}// namespace vox
