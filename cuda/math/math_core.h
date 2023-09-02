////  Copyright (c) 2023 Feng Yang
////
////  I am making my contributions/submissions to this project solely in my
////  personal capacity and am not conveying any rights to any intellectual
////  property of any third parties.
//
//#pragma once
//
//#include <cuda_runtime_api.h>
//
//// All built-in types and functions. To be compatible with runtime NVRTC compilation
//// this header must be independently compilable (i.e.: without external SDK headers)
//// to achieve this we redefine a subset of CRT functions (printf, pow, sin, cos, etc)
//
//#define CUDA_CALLABLE __host__ __device__
//#define CUDA_CALLABLE_DEVICE __device__
//
//#define RAD_TO_DEG 57.29577951308232087679
//#define DEG_TO_RAD 0.01745329251994329577
//
//namespace wp {
//
//// numeric types (used from generated kernels)
//typedef float float32;
//typedef double float64;
//
//typedef int8_t int8;
//typedef uint8_t uint8;
//
//typedef int16_t int16;
//typedef uint16_t uint16;
//
//typedef int32_t int32;
//typedef uint32_t uint32;
//
//typedef int64_t int64;
//typedef uint64_t uint64;
//
//// matches Python string type for constant strings
//typedef const char *str;
//
//struct half;
//
//CUDA_CALLABLE half float_to_half(float x);
//CUDA_CALLABLE float half_to_float(half x);
//
//struct half {
//    CUDA_CALLABLE inline half() : u(0) {}
//
//    CUDA_CALLABLE inline half(float f) {
//        *this = float_to_half(f);
//    }
//
//    unsigned short u;
//
//    CUDA_CALLABLE inline bool operator==(const half &h) const { return u == h.u; }
//    CUDA_CALLABLE inline bool operator!=(const half &h) const { return u != h.u; }
//    CUDA_CALLABLE inline bool operator>(const half &h) const { return half_to_float(*this) > half_to_float(h); }
//    CUDA_CALLABLE inline bool operator>=(const half &h) const { return half_to_float(*this) >= half_to_float(h); }
//    CUDA_CALLABLE inline bool operator<(const half &h) const { return half_to_float(*this) < half_to_float(h); }
//    CUDA_CALLABLE inline bool operator<=(const half &h) const { return half_to_float(*this) <= half_to_float(h); }
//
//    CUDA_CALLABLE inline bool operator!() const {
//        return float32(*this) == 0;
//    }
//
//    CUDA_CALLABLE inline half operator*=(const half &h) {
//        half prod = half(float32(*this) * float32(h));
//        this->u = prod.u;
//        return *this;
//    }
//
//    CUDA_CALLABLE inline half operator/=(const half &h) {
//        half quot = half(float32(*this) / float32(h));
//        this->u = quot.u;
//        return *this;
//    }
//
//    CUDA_CALLABLE inline half operator+=(const half &h) {
//        half sum = half(float32(*this) + float32(h));
//        this->u = sum.u;
//        return *this;
//    }
//
//    CUDA_CALLABLE inline half operator-=(const half &h) {
//        half diff = half(float32(*this) - float32(h));
//        this->u = diff.u;
//        return *this;
//    }
//
//    CUDA_CALLABLE inline operator float32() const { return float32(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator float64() const { return float64(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator int8() const { return int8(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator uint8() const { return uint8(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator int16() const { return int16(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator uint16() const { return uint16(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator int32() const { return int32(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator uint32() const { return uint32(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator int64() const { return int64(half_to_float(*this)); }
//    CUDA_CALLABLE inline operator uint64() const { return uint64(half_to_float(*this)); }
//};
//
//static_assert(sizeof(half) == 2, "Size of half / float16 type must be 2-bytes");
//
//typedef half float16;
//
//#if __CUDA_ARCH__
//
//CUDA_CALLABLE inline half float_to_half(float x) {
//    half h;
//    asm("{  cvt.rn.f16.f32 %0, %1;}\n"
//        : "=h"(h.u)
//        : "f"(x));
//    return h;
//}
//
//CUDA_CALLABLE inline float half_to_float(half x) {
//    float val;
//    asm("{  cvt.f32.f16 %0, %1;}\n"
//        : "=f"(val)
//        : "h"(x.u));
//    return val;
//}
//
//#else
//
//// adapted from Fabien Giesen's post: https://gist.github.com/rygorous/2156668
//inline half float_to_half(float x) {
//    union fp32 {
//        uint32 u;
//        float f;
//
//        struct
//        {
//            unsigned int mantissa : 23;
//            unsigned int exponent : 8;
//            unsigned int sign : 1;
//        };
//    };
//
//    fp32 f;
//    f.f = x;
//
//    fp32 f32infty = {255 << 23};
//    fp32 f16infty = {31 << 23};
//    fp32 magic = {15 << 23};
//    uint32 sign_mask = 0x80000000u;
//    uint32 round_mask = ~0xfffu;
//    half o;
//
//    uint32 sign = f.u & sign_mask;
//    f.u ^= sign;
//
//    // NOTE all the integer compares in this function can be safely
//    // compiled into signed compares since all operands are below
//    // 0x80000000. Important if you want fast straight SSE2 code
//    // (since there's no unsigned PCMPGTD).
//
//    if (f.u >= f32infty.u)                         // Inf or NaN (all exponent bits set)
//        o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;// NaN->qNaN and Inf->Inf
//    else                                           // (De)normalized number or zero
//    {
//        f.u &= round_mask;
//        f.f *= magic.f;
//        f.u -= round_mask;
//        if (f.u > f16infty.u) f.u = f16infty.u;// Clamp to signed infinity if overflowed
//
//        o.u = f.u >> 13;// Take the bits!
//    }
//
//    o.u |= sign >> 16;
//    return o;
//}
//
//inline float half_to_float(half h) {
//    union fp32 {
//        uint32 u;
//        float f;
//
//        struct
//        {
//            unsigned int mantissa : 23;
//            unsigned int exponent : 8;
//            unsigned int sign : 1;
//        };
//    };
//
//    static const fp32 magic = {113 << 23};
//    static const uint32 shifted_exp = 0x7c00 << 13;// exponent mask after shift
//    fp32 o;
//
//    o.u = (h.u & 0x7fff) << 13;    // exponent/mantissa bits
//    uint32 exp = shifted_exp & o.u;// just the exponent
//    o.u += (127 - 15) << 23;       // exponent adjust
//
//    // handle exponent special cases
//    if (exp == shifted_exp)     // Inf/NaN?
//        o.u += (128 - 16) << 23;// extra exp adjust
//    else if (exp == 0)          // Zero/Denormal?
//    {
//        o.u += 1 << 23;// extra exp adjust
//        o.f -= magic.f;// renormalize
//    }
//
//    o.u |= (h.u & 0x8000) << 16;// sign bit
//    return o.f;
//}
//
//#endif
//
//// BAD operator implementations for fp16 arithmetic...
//
//// negation:
//inline CUDA_CALLABLE half operator-(half a) {
//    return float_to_half(-half_to_float(a));
//}
//
//inline CUDA_CALLABLE half operator+(half a, half b) {
//    return float_to_half(half_to_float(a) + half_to_float(b));
//}
//
//inline CUDA_CALLABLE half operator-(half a, half b) {
//    return float_to_half(half_to_float(a) - half_to_float(b));
//}
//
//inline CUDA_CALLABLE half operator*(half a, half b) {
//    return float_to_half(half_to_float(a) * half_to_float(b));
//}
//
//inline CUDA_CALLABLE half operator*(half a, double b) {
//    return float_to_half(half_to_float(a) * b);
//}
//
//inline CUDA_CALLABLE half operator*(double a, half b) {
//    return float_to_half(a * half_to_float(b));
//}
//
//inline CUDA_CALLABLE half operator/(half a, half b) {
//    return float_to_half(half_to_float(a) / half_to_float(b));
//}
//
//template<typename T>
//CUDA_CALLABLE float cast_float(T x) { return (float)(x); }
//
//template<typename T>
//CUDA_CALLABLE int cast_int(T x) { return (int)(x); }
//
//#define kEps 0.0f
//
//// basic ops for integer types
//#define DECLARE_INT_OPS(T)                                                    \
//    inline CUDA_CALLABLE T mul(T a, T b) { return a * b; }                    \
//    inline CUDA_CALLABLE T div(T a, T b) { return a / b; }                    \
//    inline CUDA_CALLABLE T add(T a, T b) { return a + b; }                    \
//    inline CUDA_CALLABLE T sub(T a, T b) { return a - b; }                    \
//    inline CUDA_CALLABLE T mod(T a, T b) { return a % b; }                    \
//    inline CUDA_CALLABLE T min(T a, T b) { return a < b ? a : b; }            \
//    inline CUDA_CALLABLE T max(T a, T b) { return a > b ? a : b; }            \
//    inline CUDA_CALLABLE T clamp(T x, T a, T b) { return min(max(a, x), b); } \
//    inline CUDA_CALLABLE T floordiv(T a, T b) { return a / b; }               \
//    inline CUDA_CALLABLE T nonzero(T x) { return x == T(0) ? T(0) : T(1); }   \
//    inline CUDA_CALLABLE T sqrt(T x) { return 0; }                            \
//    inline CUDA_CALLABLE T bit_and(T a, T b) { return a & b; }                \
//    inline CUDA_CALLABLE T bit_or(T a, T b) { return a | b; }                 \
//    inline CUDA_CALLABLE T bit_xor(T a, T b) { return a ^ b; }                \
//    inline CUDA_CALLABLE T lshift(T a, T b) { return a << b; }                \
//    inline CUDA_CALLABLE T rshift(T a, T b) { return a >> b; }                \
//    inline CUDA_CALLABLE T invert(T x) { return ~x; }                         \
//    inline CUDA_CALLABLE bool isfinite(T x) { return true; }
//
//DECLARE_INT_OPS(int8)
//DECLARE_INT_OPS(int16)
//DECLARE_INT_OPS(int32)
//DECLARE_INT_OPS(int64)
//DECLARE_INT_OPS(uint8)
//DECLARE_INT_OPS(uint16)
//DECLARE_INT_OPS(uint32)
//DECLARE_INT_OPS(uint64)
//
//inline CUDA_CALLABLE int8 step(int8 x) { return x < 0 ? 1 : 0; }
//inline CUDA_CALLABLE int16 step(int16 x) { return x < 0 ? 1 : 0; }
//inline CUDA_CALLABLE int32 step(int32 x) { return x < 0 ? 1 : 0; }
//inline CUDA_CALLABLE int64 step(int64 x) { return x < 0 ? 1 : 0; }
//inline CUDA_CALLABLE uint8 step(uint8 x) { return 0; }
//inline CUDA_CALLABLE uint16 step(uint16 x) { return 0; }
//inline CUDA_CALLABLE uint32 step(uint32 x) { return 0; }
//inline CUDA_CALLABLE uint64 step(uint64 x) { return 0; }
//
//inline CUDA_CALLABLE int8 sign(int8 x) { return x < 0 ? -1 : 1; }
//inline CUDA_CALLABLE int8 sign(int16 x) { return x < 0 ? -1 : 1; }
//inline CUDA_CALLABLE int8 sign(int32 x) { return x < 0 ? -1 : 1; }
//inline CUDA_CALLABLE int8 sign(int64 x) { return x < 0 ? -1 : 1; }
//inline CUDA_CALLABLE uint8 sign(uint8 x) { return 1; }
//inline CUDA_CALLABLE uint16 sign(uint16 x) { return 1; }
//inline CUDA_CALLABLE uint32 sign(uint32 x) { return 1; }
//inline CUDA_CALLABLE uint64 sign(uint64 x) { return 1; }
//
//inline CUDA_CALLABLE void print(float16 f) {
//    printf("%g\n", half_to_float(f));
//}
//
//inline CUDA_CALLABLE void print(float f) {
//    printf("%g\n", f);
//}
//
//inline CUDA_CALLABLE void print(double f) {
//    printf("%g\n", f);
//}
//
//// basic ops for float types
//#define DECLARE_FLOAT_OPS(T)                                                  \
//    inline CUDA_CALLABLE T mul(T a, T b) { return a * b; }                    \
//    inline CUDA_CALLABLE T add(T a, T b) { return a + b; }                    \
//    inline CUDA_CALLABLE T sub(T a, T b) { return a - b; }                    \
//    inline CUDA_CALLABLE T min(T a, T b) { return a < b ? a : b; }            \
//    inline CUDA_CALLABLE T max(T a, T b) { return a > b ? a : b; }            \
//    inline CUDA_CALLABLE T sign(T x) { return x < T(0) ? -1 : 1; }            \
//    inline CUDA_CALLABLE T step(T x) { return x < T(0) ? T(1) : T(0); }       \
//    inline CUDA_CALLABLE T nonzero(T x) { return x == T(0) ? T(0) : T(1); }   \
//    inline CUDA_CALLABLE T clamp(T x, T a, T b) { return min(max(a, x), b); } \
//    inline CUDA_CALLABLE T div(T a, T b) {                                    \
//        return a / b;                                                         \
//    }
//
//DECLARE_FLOAT_OPS(float16)
//DECLARE_FLOAT_OPS(float32)
//DECLARE_FLOAT_OPS(float64)
//
//inline CUDA_CALLABLE float leaky_min(float a, float b, float r) { return min(a, b); }
//inline CUDA_CALLABLE float leaky_max(float a, float b, float r) { return max(a, b); }
//
//inline CUDA_CALLABLE float degrees(float x) { return x * RAD_TO_DEG; }
//inline CUDA_CALLABLE float radians(float x) { return x * DEG_TO_RAD; }
//
//inline CUDA_CALLABLE double degrees(double x) { return x * RAD_TO_DEG; }
//inline CUDA_CALLABLE double radians(double x) { return x * DEG_TO_RAD; }
//
//inline CUDA_CALLABLE half degrees(half x) { return x * RAD_TO_DEG; }
//inline CUDA_CALLABLE half radians(half x) { return x * DEG_TO_RAD; }
//
//template<typename C, typename T>
//CUDA_CALLABLE inline T select(const C &cond, const T &a, const T &b) {
//    // The double NOT operator !! casts to bool without compiler warnings.
//    return (!!cond) ? b : a;
//}
//
//template<typename T>
//CUDA_CALLABLE inline void copy(T &dest, const T &src) {
//    dest = src;
//}
//
//// some helpful operator overloads (just for C++ use, these are not adjointed)
//
//template<typename T>
//CUDA_CALLABLE inline T &operator+=(T &a, const T &b) {
//    a = add(a, b);
//    return a;
//}
//
//template<typename T>
//CUDA_CALLABLE inline T &operator-=(T &a, const T &b) {
//    a = sub(a, b);
//    return a;
//}
//
//template<typename T>
//CUDA_CALLABLE inline T operator+(const T &a, const T &b) { return add(a, b); }
//
//template<typename T>
//CUDA_CALLABLE inline T operator-(const T &a, const T &b) { return sub(a, b); }
//
//template<typename T>
//CUDA_CALLABLE inline T pos(const T &x) { return x; }
//
//// unary negation implemented as negative multiply, not sure the fp implications of this
//// may be better as 0.0 - x?
//template<typename T>
//CUDA_CALLABLE inline T neg(const T &x) { return T(0.0) - x; }
//
//// unary boolean negation
//template<typename T>
//CUDA_CALLABLE inline bool unot(const T &b) { return !b; }
//
//const int LAUNCH_MAX_DIMS = 4;// should match types.py
//
//struct launch_bounds_t {
//    int shape[LAUNCH_MAX_DIMS];// size of each dimension
//    int ndim;                  // number of valid dimension
//    size_t size;               // total number of threads
//};
//
//#ifdef __CUDACC__
//
//// store launch bounds in shared memory so
//// we can access them from any user func
//// this is to avoid having to explicitly
//// set another piece of __constant__ memory
//// from the host
//__shared__ launch_bounds_t s_launchBounds;
//
//__device__ inline void set_launch_bounds(const launch_bounds_t &b) {
//    if (threadIdx.x == 0)
//        s_launchBounds = b;
//
//    __syncthreads();
//}
//
//#else
//
//// for single-threaded CPU we store launch
//// bounds in static memory to share globally
//static launch_bounds_t s_launchBounds;
//static size_t s_threadIdx;
//
//inline void set_launch_bounds(const launch_bounds_t &b) {
//    s_launchBounds = b;
//}
//#endif
//
//inline CUDA_CALLABLE size_t grid_index() {
//#ifdef __CUDACC__
//    // Need to cast at least one of the variables being multiplied so that type promotion happens before the multiplication
//    size_t grid_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
//    return grid_index;
//#else
//    return s_threadIdx;
//#endif
//}
//
//inline CUDA_CALLABLE int tid() {
//    const size_t index = grid_index();
//
//    // For the 1-D tid() we need to warn the user if we're about to provide a truncated index
//    // Only do this in _DEBUG when called from device to avoid excessive register allocation
//#if defined(_DEBUG) || !defined(__CUDA_ARCH__)
//    if (index > 2147483647) {
//        printf("Warp warning: tid() is returning an overflowed int\n");
//    }
//#endif
//    return static_cast<int>(index);
//}
//
//inline CUDA_CALLABLE_DEVICE void tid(int &i, int &j) {
//    const size_t index = grid_index();
//
//    const size_t n = s_launchBounds.shape[1];
//
//    // convert to work item
//    i = index / n;
//    j = index % n;
//}
//
//inline CUDA_CALLABLE_DEVICE void tid(int &i, int &j, int &k) {
//    const size_t index = grid_index();
//
//    const size_t n = s_launchBounds.shape[1];
//    const size_t o = s_launchBounds.shape[2];
//
//    // convert to work item
//    i = index / (n * o);
//    j = index % (n * o) / o;
//    k = index % o;
//}
//
//inline CUDA_CALLABLE_DEVICE void tid(int &i, int &j, int &k, int &l) {
//    const size_t index = grid_index();
//
//    const size_t n = s_launchBounds.shape[1];
//    const size_t o = s_launchBounds.shape[2];
//    const size_t p = s_launchBounds.shape[3];
//
//    // convert to work item
//    i = index / (n * o * p);
//    j = index % (n * o * p) / (o * p);
//    k = index % (o * p) / p;
//    l = index % p;
//}
//
//template<typename T>
//inline CUDA_CALLABLE T atomic_add(T *buf, T value) {
//#if !defined(__CUDA_ARCH__)
//    T old = buf[0];
//    buf[0] += value;
//    return old;
//#else
//    return atomicAdd(buf, value);
//#endif
//}
//
//template<>
//inline CUDA_CALLABLE float16 atomic_add(float16 *buf, float16 value) {
//#if !defined(__CUDA_ARCH__)
//    float16 old = buf[0];
//    buf[0] += value;
//    return old;
//#elif defined(__clang__)// CUDA compiled by Clang
//    __half r = atomicAdd(reinterpret_cast<__half *>(buf), *reinterpret_cast<__half *>(&value));
//    return *reinterpret_cast<float16 *>(&r);
//#else                   // CUDA compiled by NVRTC
//    //return atomicAdd(buf, value);
//
//    /* Define __PTR for atomicAdd prototypes below, undef after done */
//#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
//#define __PTR "l"
//#else
//#define __PTR "r"
//#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
//
//    half r = 0.0;
//
//#if __CUDA_ARCH__ >= 700
//
//    asm volatile("{ atom.add.noftz.f16 %0,[%1],%2; }\n"
//                 : "=h"(r.u)
//                 : __PTR(buf), "h"(value.u)
//                 : "memory");
//#endif
//
//    return r;
//
//#undef __PTR
//
//#endif// CUDA compiled by NVRTC
//}
//
//// emulate atomic float max
//inline CUDA_CALLABLE float atomic_max(float *address, float val) {
//#if defined(__CUDA_ARCH__)
//    int *address_as_int = (int *)address;
//    int old = *address_as_int, assumed;
//
//    while (val > __int_as_float(old)) {
//        assumed = old;
//        old = atomicCAS(address_as_int, assumed,
//                        __float_as_int(val));
//    }
//
//    return __int_as_float(old);
//
//#else
//    float old = *address;
//    *address = max(old, val);
//    return old;
//#endif
//}
//
//// emulate atomic float min/max with atomicCAS()
//inline CUDA_CALLABLE float atomic_min(float *address, float val) {
//#if defined(__CUDA_ARCH__)
//    int *address_as_int = (int *)address;
//    int old = *address_as_int, assumed;
//
//    while (val < __int_as_float(old)) {
//        assumed = old;
//        old = atomicCAS(address_as_int, assumed,
//                        __float_as_int(val));
//    }
//
//    return __int_as_float(old);
//
//#else
//    float old = *address;
//    *address = min(old, val);
//    return old;
//#endif
//}
//
//inline CUDA_CALLABLE int atomic_max(int *address, int val) {
//#if defined(__CUDA_ARCH__)
//    return atomicMax(address, val);
//
//#else
//    int old = *address;
//    *address = max(old, val);
//    return old;
//#endif
//}
//
//// atomic int min
//inline CUDA_CALLABLE int atomic_min(int *address, int val) {
//#if defined(__CUDA_ARCH__)
//    return atomicMin(address, val);
//
//#else
//    int old = *address;
//    *address = min(old, val);
//    return old;
//#endif
//}
//
//}// namespace wp