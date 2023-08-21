//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define __GCC__ __GNUC__
#endif

#if defined(__clang__)
// CLANG ENABLE/DISABLE WARNING DEFINITION
#define VKBP_DISABLE_WARNINGS()                             \
    _Pragma("clang diagnostic push")                        \
        _Pragma("clang diagnostic ignored \"-Wall\"")       \
            _Pragma("clang diagnostic ignored \"-Wextra\"") \
                _Pragma("clang diagnostic ignored \"-Wtautological-compare\"")

#define VKBP_ENABLE_WARNINGS() \
    _Pragma("clang diagnostic pop")
#elif defined(__GNUC__) || defined(__GNUG__)
// GCC ENABLE/DISABLE WARNING DEFINITION
#define VKBP_DISABLE_WARNINGS()                             \
    _Pragma("GCC diagnostic push")                          \
        _Pragma("GCC diagnostic ignored \"-Wall\"")         \
            _Pragma("clang diagnostic ignored \"-Wextra\"") \
                _Pragma("clang diagnostic ignored \"-Wtautological-compare\"")

#define VKBP_ENABLE_WARNINGS() \
    _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
// MSVC ENABLE/DISABLE WARNING DEFINITION
#define VKBP_DISABLE_WARNINGS() \
    __pragma(warning(push, 0))

#define VKBP_ENABLE_WARNINGS() \
    __pragma(warning(pop))
#endif

VKBP_DISABLE_WARNINGS()
// TODO: replace with a direct fmt submodule
#include <spdlog/fmt/fmt.h>
VKBP_ENABLE_WARNINGS()

#include <stdexcept>

template<typename... Args>
inline void ERRORF(const std::string &format, Args &&...args) {
    throw std::runtime_error(fmt::format(format, std::forward<Args>(args)...));
}

inline void ERRORF(const std::string &message) {
    throw std::runtime_error(message);
}

#define NOT_IMPLEMENTED() ERRORF("not implemented")