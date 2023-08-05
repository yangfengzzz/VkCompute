//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "configuration.h"

namespace vox {
BoolSetting::BoolSetting(bool &handle, bool value) : handle{handle},
                                                     value{value} {
}

void BoolSetting::set() {
    handle = value;
}

std::type_index BoolSetting::get_type() {
    return typeid(BoolSetting);
}

IntSetting::IntSetting(int &handle, int value) : handle{handle},
                                                 value{value} {
}

void IntSetting::set() {
    handle = value;
}

std::type_index IntSetting::get_type() {
    return typeid(IntSetting);
}

EmptySetting::EmptySetting() {
}

void EmptySetting::set() {
}

std::type_index EmptySetting::get_type() {
    return typeid(EmptySetting);
}

void Configuration::set() {
    for (auto pair : current_configuration->second) {
        for (auto setting : pair.second) {
            setting->set();
        }
    }
}

bool Configuration::next() {
    if (configs.size() == 0) {
        return false;
    }

    current_configuration++;

    if (current_configuration == configs.end()) {
        return false;
    }

    return true;
}

void Configuration::reset() {
    current_configuration = configs.begin();
}

void Configuration::insert_setting(uint32_t config_index, std::unique_ptr<Setting> setting) {
    settings.push_back(std::move(setting));
    configs[config_index][settings.back()->get_type()].push_back(settings.back().get());
}

}// namespace vox
