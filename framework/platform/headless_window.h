//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "platform/window.h"

namespace vox {
/**
 * @brief Surface-less implementation of a Window, for use in headless rendering
 */
class HeadlessWindow : public Window {
public:
    HeadlessWindow(const Window::Properties &properties);

    virtual ~HeadlessWindow() = default;

    /**
	 * @brief A direct window doesn't have a surface
	 * @returns VK_NULL_HANDLE
	 */
    VkSurfaceKHR create_surface(Instance &instance) override;

    /**
	 * @brief A direct window doesn't have a surface
	 * @returns nullptr
	 */
    VkSurfaceKHR create_surface(VkInstance instance, VkPhysicalDevice physical_device) override;

    bool should_close() override;

    void close() override;

    float get_dpi_factor() const override;

    std::vector<const char *> get_required_surface_extensions() const override;

private:
    bool closed{false};
};
}// namespace vox
