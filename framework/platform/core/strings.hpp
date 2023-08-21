//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

namespace vox {
/**
 * @brief Replaces all occurrences of a substring with another substring.
 */
std::string replace_all(std::string str, const std::string &from, const std::string &to);

/**
 * @brief Removes all occurrences of a set of characters from the end of a string.
 */
std::string trim_right(const std::string &str, const std::string &chars = " ");

/**
 * @brief Removes all occurrences of a set of characters from the beginning of a string.
 */
std::string trim_left(const std::string &str, const std::string &chars = " ");
}// namespace vox