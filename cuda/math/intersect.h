//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "vec.h"
#include <cuda.h>

namespace wp {

__device__ inline vec3 closest_point_to_aabb(const vec3 &p, const vec3 &lower, const vec3 &upper) {
    vec3 c;

    {
        float v = p[0];
        if (v < lower[0]) v = lower[0];
        if (v > upper[0]) v = upper[0];
        c[0] = v;
    }

    {
        float v = p[1];
        if (v < lower[1]) v = lower[1];
        if (v > upper[1]) v = upper[1];
        c[1] = v;
    }

    {
        float v = p[2];
        if (v < lower[2]) v = lower[2];
        if (v > upper[2]) v = upper[2];
        c[2] = v;
    }

    return c;
}

__device__ inline vec2 closest_point_to_triangle(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &p) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ap = p - a;

    float u, v, w;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        v = 0.0f;
        w = 0.0f;
        u = 1.0f - v - w;
        return vec2(u, v);
    }

    vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        v = 1.0f;
        w = 0.0f;
        u = 1.0f - v - w;
        return vec2(u, v);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        v = d1 / (d1 - d3);
        w = 0.0f;
        u = 1.0f - v - w;
        return vec2(u, v);
    }

    vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        v = 0.0f;
        w = 1.0f;
        u = 1.0f - v - w;
        return vec2(u, v);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        v = 0.0f;
        w = d2 / (d2 - d6);
        u = 1.0f - v - w;
        return vec2(u, v);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        v = 1.0f - w;
        u = 1.0f - v - w;
        return vec2(u, v);
    }

    float denom = 1.0f / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    u = 1.0f - v - w;
    return vec2(u, v);
}

__device__ inline bool intersect_ray_aabb(const vec3 &pos, const vec3 &rcp_dir, const vec3 &lower, const vec3 &upper, float &t) {
    float l1, l2, lmin, lmax;

    l1 = (lower[0] - pos[0]) * rcp_dir[0];
    l2 = (upper[0] - pos[0]) * rcp_dir[0];
    lmin = min(l1, l2);
    lmax = max(l1, l2);

    l1 = (lower[1] - pos[1]) * rcp_dir[1];
    l2 = (upper[1] - pos[1]) * rcp_dir[1];
    lmin = max(min(l1, l2), lmin);
    lmax = min(max(l1, l2), lmax);

    l1 = (lower[2] - pos[2]) * rcp_dir[2];
    l2 = (upper[2] - pos[2]) * rcp_dir[2];
    lmin = max(min(l1, l2), lmin);
    lmax = min(max(l1, l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
}

// Moller and Trumbore's method
__device__ inline bool intersect_ray_tri_moller(const vec3 &p, const vec3 &dir, const vec3 &a, const vec3 &b, const vec3 &c, float &t, float &u, float &v, float &w, float &sign, vec3 *normal) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 n = cross(ab, ac);

    float d = dot(-dir, n);
    float ood = 1.0f / d;// No need to check for division by zero here as infinity arithmetic will save us...
    vec3 ap = p - a;

    t = dot(ap, n) * ood;
    if (t < 0.0f)
        return false;

    vec3 e = cross(-dir, ap);
    v = dot(ac, e) * ood;
    if (v < 0.0f || v > 1.0f)// ...here...
        return false;
    w = -dot(ab, e) * ood;
    if (w < 0.0f || (v + w) > 1.0f)// ...and here
        return false;

    u = 1.0f - v - w;
    if (normal)
        *normal = n;

    sign = d;

    return true;
}

__device__ inline bool intersect_ray_tri_rtcd(const vec3 &p, const vec3 &dir, const vec3 &a, const vec3 &b, const vec3 &c, float &t, float &u, float &v, float &w, float &sign, vec3 *normal) {
    const vec3 ab = b - a;
    const vec3 ac = c - a;

    // calculate normal
    vec3 n = cross(ab, ac);

    // need to solve a system of three equations to give t, u, v
    float d = dot(-dir, n);

    // if dir is parallel to triangle plane or points away from triangle
    if (d <= 0.0f)
        return false;

    vec3 ap = p - a;
    t = dot(ap, n);

    // ignores tris behind
    if (t < 0.0f)
        return false;

    // compute barycentric coordinates
    vec3 e = cross(-dir, ap);
    v = dot(ac, e);
    if (v < 0.0f || v > d) return false;

    w = -dot(ab, e);
    if (w < 0.0f || v + w > d) return false;

    float ood = 1.0f / d;
    t *= ood;
    v *= ood;
    w *= ood;
    u = 1.0f - v - w;

    // optionally write out normal (todo: this branch is a performance concern, should probably remove)
    if (normal)
        *normal = n;

    return true;
}

#ifndef __CUDA_ARCH__

// these are provided as built-ins by CUDA
inline float __int_as_float(int i) {
    return *(float *)(&i);
}

inline int __float_as_int(float f) {
    return *(int *)(&f);
}

#endif

__device__ inline float xorf(float x, int y) {
    return __int_as_float(__float_as_int(x) ^ y);
}

__device__ inline int sign_mask(float x) {
    return __float_as_int(x) & 0x80000000;
}

__device__ inline int max_dim(vec3 a) {
    float x = abs(a[0]);
    float y = abs(a[1]);
    float z = abs(a[2]);

    return longest_axis(vec3(x, y, z));
}

// computes the difference of products a*b - c*d using
// FMA instructions for improved numerical precision
__device__ inline float diff_product(float a, float b, float c, float d) {
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}

// http://jcgt.org/published/0002/01/05/
__device__ inline bool intersect_ray_tri_woop(const vec3 &p, const vec3 &dir, const vec3 &a, const vec3 &b, const vec3 &c, float &t, float &u, float &v, float &sign, vec3 *normal) {
    // todo: precompute for ray

    int kz = max_dim(dir);
    int kx = kz + 1;
    if (kx == 3) kx = 0;
    int ky = kx + 1;
    if (ky == 3) ky = 0;

    if (dir[kz] < 0.0f) {
        float tmp = kx;
        kx = ky;
        ky = tmp;
    }

    float Sx = dir[kx] / dir[kz];
    float Sy = dir[ky] / dir[kz];
    float Sz = 1.0f / dir[kz];

    // todo: end precompute

    const vec3 A = a - p;
    const vec3 B = b - p;
    const vec3 C = c - p;

    const float Ax = A[kx] - Sx * A[kz];
    const float Ay = A[ky] - Sy * A[kz];
    const float Bx = B[kx] - Sx * B[kz];
    const float By = B[ky] - Sy * B[kz];
    const float Cx = C[kx] - Sx * C[kz];
    const float Cy = C[ky] - Sy * C[kz];

    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

    if (U == 0.0f || V == 0.0f || W == 0.0f) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) {
        return false;
    }

    float det = U + V + W;

    if (det == 0.0f) {
        return false;
    }

    const float Az = Sz * A[kz];
    const float Bz = Sz * B[kz];
    const float Cz = Sz * C[kz];
    const float T = U * Az + V * Bz + W * Cz;

    int det_sign = sign_mask(det);
    if (xorf(T, det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
        return false;
    }

    const float rcpDet = 1.0f / det;
    u = U * rcpDet;
    v = V * rcpDet;
    t = T * rcpDet;
    sign = det;

    // optionally write out normal (todo: this branch is a performance concern, should probably remove)
    if (normal) {
        const vec3 ab = b - a;
        const vec3 ac = c - a;

        // calculate normal
        *normal = cross(ab, ac);
    }

    return true;
}

// Möller's method
#include "intersect_tri.h"

__device__ inline int intersect_tri_tri(
    vec3 &v0, vec3 &v1, vec3 &v2,
    vec3 &u0, vec3 &u1, vec3 &u2) {
    return NoDivTriTriIsect(&v0[0], &v1[0], &v2[0], &u0[0], &u1[0], &u2[0]);
}

static __device__ vec3 closest_point_edge_edge(vec3 var_p1,
                                                  vec3 var_q1,
                                                  vec3 var_p2,
                                                  vec3 var_q2,
                                                  float32 var_epsilon) {
    //---------
    // primal vars
    vec3 var_0;
    vec3 var_1;
    vec3 var_2;
    float32 var_3;
    float32 var_4;
    float32 var_5;
    const float32 var_6 = 0.0;
    float32 var_7;
    float32 var_8;
    vec3 var_9;
    float32 var_10;
    bool var_11;
    bool var_12;
    bool var_13;
    vec3 var_14;
    bool var_15;
    float32 var_16;
    float32 var_17;
    float32 var_18;
    float32 var_19;
    float32 var_20;
    float32 var_21;
    bool var_22;
    float32 var_23;
    float32 var_24;
    const float32 var_25 = 1.0;
    float32 var_26;
    float32 var_27;
    float32 var_28;
    float32 var_29;
    float32 var_30;
    float32 var_31;
    float32 var_32;
    float32 var_33;
    bool var_34;
    float32 var_35;
    float32 var_36;
    float32 var_37;
    float32 var_38;
    float32 var_39;
    float32 var_40;
    float32 var_41;
    float32 var_42;
    float32 var_43;
    float32 var_44;
    bool var_45;
    float32 var_46;
    float32 var_47;
    float32 var_48;
    float32 var_49;
    float32 var_50;
    bool var_51;
    float32 var_52;
    float32 var_53;
    float32 var_54;
    float32 var_55;
    float32 var_56;
    float32 var_57;
    float32 var_58;
    float32 var_59;
    float32 var_60;
    float32 var_61;
    float32 var_62;
    vec3 var_63;
    vec3 var_64;
    vec3 var_65;
    vec3 var_66;
    vec3 var_67;
    vec3 var_68;
    vec3 var_69;
    float32 var_70;
    vec3 var_71;
    //---------
    // forward
    var_0 = wp::sub(var_q1, var_p1);
    var_1 = wp::sub(var_q2, var_p2);
    var_2 = wp::sub(var_p1, var_p2);
    var_3 = wp::dot(var_0, var_0);
    var_4 = wp::dot(var_1, var_1);
    var_5 = wp::dot(var_1, var_2);
    var_7 = wp::cast_float(var_6);
    var_8 = wp::cast_float(var_6);
    var_9 = wp::sub(var_p2, var_p1);
    var_10 = wp::length(var_9);
    var_11 = (var_3 <= var_epsilon);
    var_12 = (var_4 <= var_epsilon);
    var_13 = var_11 && var_12;
    if (var_13) {
        var_14 = wp::vec3(var_7, var_8, var_10);
        return var_14;
    }
    var_15 = (var_3 <= var_epsilon);
    if (var_15) {
        var_16 = wp::cast_float(var_6);
        var_17 = wp::div(var_5, var_4);
        var_18 = wp::cast_float(var_17);
    }
    var_19 = wp::select(var_15, var_7, var_16);
    var_20 = wp::select(var_15, var_8, var_18);
    if (!var_15) {
        var_21 = wp::dot(var_0, var_2);
        var_22 = (var_4 <= var_epsilon);
        if (var_22) {
            var_23 = wp::neg(var_21);
            var_24 = wp::div(var_23, var_3);
            var_26 = wp::clamp(var_24, var_6, var_25);
            var_27 = wp::cast_float(var_6);
        }
        var_28 = wp::select(var_22, var_19, var_26);
        var_29 = wp::select(var_22, var_20, var_27);
        if (!var_22) {
            var_30 = wp::dot(var_0, var_1);
            var_31 = wp::mul(var_3, var_4);
            var_32 = wp::mul(var_30, var_30);
            var_33 = wp::sub(var_31, var_32);
            var_34 = (var_33 != var_6);
            if (var_34) {
                var_35 = wp::mul(var_30, var_5);
                var_36 = wp::mul(var_21, var_4);
                var_37 = wp::sub(var_35, var_36);
                var_38 = wp::div(var_37, var_33);
                var_39 = wp::clamp(var_38, var_6, var_25);
            }
            var_40 = wp::select(var_34, var_28, var_39);
            if (!var_34) {
            }
            var_41 = wp::select(var_34, var_6, var_40);
            var_42 = wp::mul(var_30, var_41);
            var_43 = wp::add(var_42, var_5);
            var_44 = wp::div(var_43, var_4);
            var_45 = (var_44 < var_6);
            if (var_45) {
                var_46 = wp::neg(var_21);
                var_47 = wp::div(var_46, var_3);
                var_48 = wp::clamp(var_47, var_6, var_25);
            }
            var_49 = wp::select(var_45, var_41, var_48);
            var_50 = wp::select(var_45, var_44, var_6);
            if (!var_45) {
                var_51 = (var_50 > var_25);
                if (var_51) {
                    var_52 = wp::sub(var_30, var_21);
                    var_53 = wp::div(var_52, var_3);
                    var_54 = wp::clamp(var_53, var_6, var_25);
                }
                var_55 = wp::select(var_51, var_49, var_54);
                var_56 = wp::select(var_51, var_50, var_25);
            }
            var_57 = wp::select(var_45, var_55, var_49);
            var_58 = wp::select(var_45, var_56, var_50);
        }
        var_59 = wp::select(var_22, var_57, var_28);
        var_60 = wp::select(var_22, var_58, var_29);
    }
    var_61 = wp::select(var_15, var_59, var_19);
    var_62 = wp::select(var_15, var_60, var_20);
    var_63 = wp::sub(var_q1, var_p1);
    var_64 = wp::mul(var_63, var_61);
    var_65 = wp::add(var_p1, var_64);
    var_66 = wp::sub(var_q2, var_p2);
    var_67 = wp::mul(var_66, var_62);
    var_68 = wp::add(var_p2, var_67);
    var_69 = wp::sub(var_68, var_65);
    var_70 = wp::length(var_69);
    var_71 = wp::vec3(var_61, var_62, var_70);
    return var_71;
}
}// namespace wp
