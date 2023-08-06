//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/utils.h"
#include "common/vk_common.h"
#include "platform/window.h"
#include "platform/input_events.h"
#include "../timer.h"

#if defined(VK_USE_PLATFORM_XLIB_KHR)
#undef Success
#endif

namespace vox {
enum class ExitCode {
    Success = 0, /* App executed as expected */
    Help,        /* App should show help */
    Close,       /* App has been requested to close at initialization */
    FatalError   /* App encountered an unexpected error */
};

class Platform {
public:
    Platform() = default;

    virtual ~Platform() = default;

    /**
	 * @brief Initialize the platform
	 * @param plugins plugins available to the platform
	 * @return An exit code representing the outcome of initialization
	 */
    virtual ExitCode initialize(std::function<void(float)> update_callback,
                                std::function<void(uint32_t, uint32_t)> resize_callback = {},
                                std::function<void(const InputEvent &)> event_callback = {});

    /**
	 * @brief Handles the main loop of the platform
	 * This should be overriden if a platform requires a specific main loop setup.
	 * @return An exit code representing the outcome of the loop
	 */
    ExitCode main_loop();

    /**
	 * @brief Runs the application for one frame
	 */
    void update();

    /**
	 * @brief Terminates the platform and the application
	 * @param code Determines how the platform should exit
	 */
    virtual void terminate(ExitCode code);

    /**
	 * @brief Requests to close the platform at the next available point
	 */
    virtual void close();

public:
    Window &get_window();

    virtual void resize(uint32_t width, uint32_t height);

    virtual void input_event(const InputEvent &input_event);

    void set_focus(bool focused);

    void force_simulation_fps(float fps);

    void disable_input_processing();

    void set_window_properties(const Window::OptionalProperties &properties);

    static const uint32_t MIN_WINDOW_WIDTH;
    static const uint32_t MIN_WINDOW_HEIGHT;

protected:
    std::unique_ptr<Window> window{nullptr};

    virtual std::vector<spdlog::sink_ptr> get_platform_sinks();

    /**
	 * @brief Handles the creation of the window
	 *
	 * @param properties Preferred window configuration
	 */
    virtual void create_window(const Window::Properties &properties) = 0;

    Window::Properties window_properties; /* Source of truth for window state */
    bool fixed_simulation_fps{false};     /* Delta time should be fixed with a fabricated value */
    float simulation_frame_time = 0.016f; /* A fabricated delta time */
    bool process_input_events{true};      /* App should continue processing input events */
    bool focused{true};                   /* App is currently in focus at an operating system level */
    bool close_requested{false};          /* Close requested */

    std::function<void(float)> update_callback;
    std::function<void(uint32_t, uint32_t)> resize_callback;
    std::function<void(const InputEvent &)> event_callback;

private:
    Timer timer;
};

}// namespace vox
