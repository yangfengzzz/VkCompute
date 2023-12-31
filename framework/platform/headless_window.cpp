//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "headless_window.h"

namespace vox {
HeadlessWindow::HeadlessWindow(const Window::Properties &properties) : Window(properties) {
}

VkSurfaceKHR HeadlessWindow::create_surface(core::Instance &instance) {
    return VK_NULL_HANDLE;
}

VkSurfaceKHR HeadlessWindow::create_surface(VkInstance, VkPhysicalDevice) {
    return VK_NULL_HANDLE;
}

bool HeadlessWindow::should_close() {
    return closed;
}

void HeadlessWindow::close() {
    closed = true;
}

float HeadlessWindow::get_dpi_factor() const {
    // This factor is used for scaling UI elements, so return 1.0f (1 x n = n)
    return 1.0f;
}

std::vector<const char *> HeadlessWindow::get_required_surface_extensions() const {
    return {};
}
}// namespace vox
