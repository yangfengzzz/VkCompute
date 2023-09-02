////  Copyright (c) 2023 Feng Yang
////
////  I am making my contributions/submissions to this project solely in my
////  personal capacity and am not conveying any rights to any intellectual
////  property of any third parties.
//
//#pragma once
//
//#include "math/math_core.h"
//#include "math/vec.h"
//#include "math/mat.h"
//#include "math/quat.h"
//#include "math/spatial.h"
//#include "math/intersect.h"
//
////--------------
//namespace wp {
//
//// dot for scalar types just to make some templates compile for scalar/vector
//inline CUDA_CALLABLE float dot(float a, float b) { return mul(a, b); }
//inline CUDA_CALLABLE float tensordot(float a, float b) { return mul(a, b); }
//
//#define DECLARE_INTERP_FUNCS(T)                                                                                        \
//    CUDA_CALLABLE inline T smoothstep(T edge0, T edge1, T x) {                                                         \
//        x = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));                                                          \
//        return x * x * (T(3) - T(2) * x);                                                                              \
//    }                                                                                                                  \
//    CUDA_CALLABLE inline T lerp(const T &a, const T &b, T t) {                                                         \
//        return a * (T(1) - t) + b * t;                                                                                 \
//    }
//
//DECLARE_INTERP_FUNCS(float16)
//DECLARE_INTERP_FUNCS(float32)
//DECLARE_INTERP_FUNCS(float64)
//
//inline CUDA_CALLABLE void print(const str s) {
//    printf("%s\n", s);
//}
//
//inline CUDA_CALLABLE void print(int i) {
//    printf("%d\n", i);
//}
//
//inline CUDA_CALLABLE void print(short i) {
//    printf("%hd\n", i);
//}
//
//inline CUDA_CALLABLE void print(long i) {
//    printf("%ld\n", i);
//}
//
//inline CUDA_CALLABLE void print(long long i) {
//    printf("%lld\n", i);
//}
//
//inline CUDA_CALLABLE void print(unsigned i) {
//    printf("%u\n", i);
//}
//
//inline CUDA_CALLABLE void print(unsigned short i) {
//    printf("%hu\n", i);
//}
//
//inline CUDA_CALLABLE void print(unsigned long i) {
//    printf("%lu\n", i);
//}
//
//inline CUDA_CALLABLE void print(unsigned long long i) {
//    printf("%llu\n", i);
//}
//
//template<unsigned Length, typename Type>
//inline CUDA_CALLABLE void print(vec_t<Length, Type> v) {
//    for (unsigned i = 0; i < Length; ++i) {
//        printf("%g ", float(v[i]));
//    }
//    printf("\n");
//}
//
//template<typename Type>
//inline CUDA_CALLABLE void print(quat_t<Type> i) {
//    printf("%g %g %g %g\n", float(i.x), float(i.y), float(i.z), float(i.w));
//}
//
//template<unsigned Rows, unsigned Cols, typename Type>
//inline CUDA_CALLABLE void print(const mat_t<Rows, Cols, Type> &m) {
//    for (unsigned i = 0; i < Rows; ++i) {
//        for (unsigned j = 0; j < Cols; ++j) {
//            printf("%g ", float(m.data[i][j]));
//        }
//        printf("\n");
//    }
//}
//
//template<typename Type>
//inline CUDA_CALLABLE void print(transform_t<Type> t) {
//    printf("(%g %g %g) (%g %g %g %g)\n", float(t.p[0]), float(t.p[1]), float(t.p[2]), float(t.q.x), float(t.q.y), float(t.q.z), float(t.q.w));
//}
//
//template<typename T>
//inline CUDA_CALLABLE void expect_eq(const T &actual, const T &expected) {
//    if (!(actual == expected)) {
//        printf("Error, expect_eq() failed:\n");
//        printf("\t Expected: ");
//        print(expected);
//        printf("\t Actual: ");
//        print(actual);
//    }
//}
//
//template<typename T>
//inline CUDA_CALLABLE void expect_neq(const T &actual, const T &expected) {
//    if (actual == expected) {
//        printf("Error, expect_neq() failed:\n");
//        printf("\t Expected: ");
//        print(expected);
//        printf("\t Actual: ");
//        print(actual);
//    }
//}
//
//template<typename T>
//inline CUDA_CALLABLE void expect_near(const T &actual, const T &expected, const T &tolerance) {
//    if (abs(actual - expected) > tolerance) {
//        printf("Error, expect_near() failed with tolerance ");
//        print(tolerance);
//        printf("\t Expected: ");
//        print(expected);
//        printf("\t Actual: ");
//        print(actual);
//    }
//}
//
//inline CUDA_CALLABLE void expect_near(const vec3 &actual, const vec3 &expected, const float &tolerance) {
//    const float diff = max(max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])), abs(actual[2] - expected[2]));
//    if (diff > tolerance) {
//        printf("Error, expect_near() failed with tolerance ");
//        print(tolerance);
//        printf("\t Expected: ");
//        print(expected);
//        printf("\t Actual: ");
//        print(actual);
//    }
//}
//
//}// namespace wp
//
//// include array.h so we have the print, isfinite functions for the inner array types defined
//#include "math/array.h"
//#include "structure/mesh.h"
//#include "structure/bvh.h"
//#include "math/svd.h"
//#include "structure/hashgrid.h"
//#include "math/range.h"
//#include "math/rand.h"
//#include "math/noise.h"
//#include "math/matnn.h"
