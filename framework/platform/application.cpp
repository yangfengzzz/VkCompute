//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "application.h"

#include "common/logging.h"
#include "platform/window.h"

namespace vox {
Application::Application() : name{"Sample Name"} {
}

bool Application::prepare(const ApplicationOptions &options) {
    assert(options.window != nullptr && "Window must be valid");

    auto &_debug_info = get_debug_info();
    _debug_info.insert<field::MinMax, float>("fps", fps);
    _debug_info.insert<field::MinMax, float>("frame_time", frame_time);

    lock_simulation_speed = options.benchmark_enabled;
    window = options.window;

    return true;
}

void Application::finish() {
}

bool Application::resize(const uint32_t /*width*/, const uint32_t /*height*/) {
    return true;
}

void Application::input_event(const InputEvent &input_event) {
}

void Application::update(float delta_time) {
    fps = 1.0f / delta_time;
    frame_time = delta_time * 1000.0f;
}

const std::string &Application::get_name() const {
    return name;
}

void Application::set_name(const std::string &name_) {
    name = name_;
}

DebugInfo &Application::get_debug_info() {
    return debug_info;
}
}// namespace vox
