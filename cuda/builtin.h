//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/math_core.h"
#include "math/vec.h"
#include "math/mat.h"
#include "math/quat.h"
#include "math/spatial.h"
#include "math/intersect.h"
#include "math/intersect_adj.h"

//--------------
namespace wp {

// dot for scalar types just to make some templates compile for scalar/vector
inline CUDA_CALLABLE float dot(float a, float b) { return mul(a, b); }
inline CUDA_CALLABLE void adj_dot(float a, float b, float &adj_a, float &adj_b, float adj_ret) { adj_mul(a, b, adj_a, adj_b, adj_ret); }
inline CUDA_CALLABLE float tensordot(float a, float b) { return mul(a, b); }

#define DECLARE_INTERP_FUNCS(T)                                                                                        \
    CUDA_CALLABLE inline T smoothstep(T edge0, T edge1, T x) {                                                         \
        x = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));                                                          \
        return x * x * (T(3) - T(2) * x);                                                                              \
    }                                                                                                                  \
    CUDA_CALLABLE inline void adj_smoothstep(T edge0, T edge1, T x, T &adj_edge0, T &adj_edge1, T &adj_x, T adj_ret) { \
        T ab = edge0 - edge1;                                                                                          \
        T ax = edge0 - x;                                                                                              \
        T bx = edge1 - x;                                                                                              \
        T xb = x - edge1;                                                                                              \
                                                                                                                       \
        if (bx / ab >= T(0) || ax / ab <= T(0)) {                                                                      \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        T ab3 = ab * ab * ab;                                                                                          \
        T ab4 = ab3 * ab;                                                                                              \
        adj_edge0 += adj_ret * ((T(6) * ax * bx * bx) / ab4);                                                          \
        adj_edge1 += adj_ret * ((T(6) * ax * ax * xb) / ab4);                                                          \
        adj_x += adj_ret * ((T(6) * ax * bx) / ab3);                                                                   \
    }                                                                                                                  \
    CUDA_CALLABLE inline T lerp(const T &a, const T &b, T t) {                                                         \
        return a * (T(1) - t) + b * t;                                                                                 \
    }                                                                                                                  \
    CUDA_CALLABLE inline void adj_lerp(const T &a, const T &b, T t, T &adj_a, T &adj_b, T &adj_t, const T &adj_ret) {  \
        adj_a += adj_ret * (T(1) - t);                                                                                 \
        adj_b += adj_ret * t;                                                                                          \
        adj_t += b * adj_ret - a * adj_ret;                                                                            \
    }

DECLARE_INTERP_FUNCS(float16)
DECLARE_INTERP_FUNCS(float32)
DECLARE_INTERP_FUNCS(float64)

inline CUDA_CALLABLE void print(const str s) {
    printf("%s\n", s);
}

inline CUDA_CALLABLE void print(int i) {
    printf("%d\n", i);
}

inline CUDA_CALLABLE void print(short i) {
    printf("%hd\n", i);
}

inline CUDA_CALLABLE void print(long i) {
    printf("%ld\n", i);
}

inline CUDA_CALLABLE void print(long long i) {
    printf("%lld\n", i);
}

inline CUDA_CALLABLE void print(unsigned i) {
    printf("%u\n", i);
}

inline CUDA_CALLABLE void print(unsigned short i) {
    printf("%hu\n", i);
}

inline CUDA_CALLABLE void print(unsigned long i) {
    printf("%lu\n", i);
}

inline CUDA_CALLABLE void print(unsigned long long i) {
    printf("%llu\n", i);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void print(vec_t<Length, Type> v) {
    for (unsigned i = 0; i < Length; ++i) {
        printf("%g ", float(v[i]));
    }
    printf("\n");
}

template<typename Type>
inline CUDA_CALLABLE void print(quat_t<Type> i) {
    printf("%g %g %g %g\n", float(i.x), float(i.y), float(i.z), float(i.w));
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void print(const mat_t<Rows, Cols, Type> &m) {
    for (unsigned i = 0; i < Rows; ++i) {
        for (unsigned j = 0; j < Cols; ++j) {
            printf("%g ", float(m.data[i][j]));
        }
        printf("\n");
    }
}

template<typename Type>
inline CUDA_CALLABLE void print(transform_t<Type> t) {
    printf("(%g %g %g) (%g %g %g %g)\n", float(t.p[0]), float(t.p[1]), float(t.p[2]), float(t.q.x), float(t.q.y), float(t.q.z), float(t.q.w));
}

inline CUDA_CALLABLE void adj_print(int i, int adj_i) { printf("%d adj: %d\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(float f, float adj_f) { printf("%g adj: %g\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(short f, short adj_f) { printf("%hd adj: %hd\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(long f, long adj_f) { printf("%ld adj: %ld\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(long long f, long long adj_f) { printf("%lld adj: %lld\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned f, unsigned adj_f) { printf("%u adj: %u\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned short f, unsigned short adj_f) { printf("%hu adj: %hu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned long f, unsigned long adj_f) { printf("%lu adj: %lu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned long long f, unsigned long long adj_f) { printf("%llu adj: %llu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(half h, half adj_h) { printf("%g adj: %g\n", half_to_float(h), half_to_float(adj_h)); }
inline CUDA_CALLABLE void adj_print(double f, double adj_f) { printf("%g adj: %g\n", f, adj_f); }

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_print(vec_t<Length, Type> v, vec_t<Length, Type> &adj_v) { printf("%g %g adj: %g %g \n", v[0], v[1], adj_v[0], adj_v[1]); }

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_print(mat_t<Rows, Cols, Type> m, mat_t<Rows, Cols, Type> &adj_m) {}

template<typename Type>
inline CUDA_CALLABLE void adj_print(quat_t<Type> q, quat_t<Type> &adj_q) { printf("%g %g %g %g adj: %g %g %g %g\n", q.x, q.y, q.z, q.w, adj_q.x, adj_q.y, adj_q.z, adj_q.w); }

template<typename Type>
inline CUDA_CALLABLE void adj_print(transform_t<Type> t, transform_t<Type> &adj_t) {}

inline CUDA_CALLABLE void adj_print(str t, str &adj_t) {}

// printf defined globally in crt.h
inline CUDA_CALLABLE void adj_printf(const char *fmt, ...) {}

template<typename T>
inline CUDA_CALLABLE void expect_eq(const T &actual, const T &expected) {
    if (!(actual == expected)) {
        printf("Error, expect_eq() failed:\n");
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

template<typename T>
inline CUDA_CALLABLE void adj_expect_eq(const T &a, const T &b, T &adj_a, T &adj_b) {
    // nop
}

template<typename T>
inline CUDA_CALLABLE void expect_neq(const T &actual, const T &expected) {
    if (actual == expected) {
        printf("Error, expect_neq() failed:\n");
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

template<typename T>
inline CUDA_CALLABLE void adj_expect_neq(const T &a, const T &b, T &adj_a, T &adj_b) {
    // nop
}

template<typename T>
inline CUDA_CALLABLE void expect_near(const T &actual, const T &expected, const T &tolerance) {
    if (abs(actual - expected) > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        print(tolerance);
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

inline CUDA_CALLABLE void expect_near(const vec3 &actual, const vec3 &expected, const float &tolerance) {
    const float diff = max(max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])), abs(actual[2] - expected[2]));
    if (diff > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        print(tolerance);
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

template<typename T>
inline CUDA_CALLABLE void adj_expect_near(const T &actual, const T &expected, const T &tolerance, T &adj_actual, T &adj_expected, T &adj_tolerance) {
    // nop
}

inline CUDA_CALLABLE void adj_expect_near(const vec3 &actual, const vec3 &expected, float tolerance, vec3 &adj_actual, vec3 &adj_expected, float adj_tolerance) {
    // nop
}

}// namespace wp

// include array.h so we have the print, isfinite functions for the inner array types defined
#include "math/array.h"
#include "structure/mesh.h"
#include "structure/bvh.h"
#include "math/svd.h"
#include "structure/hashgrid.h"
#include "math/range.h"
#include "math/rand.h"
#include "math/noise.h"
#include "math/matnn.h"
