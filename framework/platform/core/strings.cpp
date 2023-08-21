//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "platform/core/strings.hpp"

namespace vox {
std::string replace_all(std::string str, const std::string &from, const std::string &to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length() - 1;
    }
    return str;
}

std::string trim_right(const std::string &str, const std::string &chars) {
    std::string result = str;
    result.erase(str.find_last_not_of(chars) + 1);
    return result;
}

std::string trim_left(const std::string &str, const std::string &chars) {
    std::string result = str;
    result.erase(0, str.find_first_not_of(chars));
    return result;
}
}// namespace vox