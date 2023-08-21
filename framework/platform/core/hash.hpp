//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <functional>

namespace vox {
inline void hash_combine(size_t &seed, size_t hash) {
    hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash;
}

/**
 * @brief Helper function to combine a given hash
 *        with a generated hash for the input param.
 */
template<class T>
inline void hash_combine(size_t &seed, const T &v) {
    std::hash<T> hasher;

    hash_combine(seed, hasher(v));
}
}// namespace vox