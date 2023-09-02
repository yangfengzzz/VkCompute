//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda_runtime_api.h>

// basic ops for float types
#define DECLARE_FLOAT_OPS(T)                                                      \
    inline __device__ __host__ T mul(T a, T b) { return a * b; }                  \
    inline __device__ __host__ T add(T a, T b) { return a + b; }                  \
    inline __device__ __host__ T sub(T a, T b) { return a - b; }                  \
    inline __device__ __host__ T sign(T x) { return x < T(0) ? -1 : 1; }          \
    inline __device__ __host__ T step(T x) { return x < T(0) ? T(1) : T(0); }     \
    inline __device__ __host__ T nonzero(T x) { return x == T(0) ? T(0) : T(1); } \
    inline __device__ __host__ T clamp(T x, T a, T b) { return min(max(a, x), b); }

DECLARE_FLOAT_OPS(float)
DECLARE_FLOAT_OPS(double)
