//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/functors.h"

namespace vox {

// MARK: - VectorExpression

//!
//! \brief Base class for vector expression.
//!
//! Vector expression is a meta type that enables template expression pattern.
//!
//! \tparam T - Real number type.
//! \tparam E - Subclass type.
//!
template <typename T, typename E>
class VectorExpression {
public:
    //! Size of the vector.
    [[nodiscard]] size_t size() const;

    //! Returns actual implementation (the subclass).
    const E &operator()() const;
};

// MARK: - VectorUnaryOp

//!
//! \brief Vector expression for unary operation.
//!
//! This matrix expression represents an unary vector operation that takes
//! single input vector expression.
//!
//! \tparam T - Real number type.
//! \tparam E - Input expression type.
//! \tparam Op - Unary operation.
//!
template <typename T, typename E, typename Op>
class VectorUnaryOp : public VectorExpression<T, VectorUnaryOp<T, E, Op>> {
public:
    //! Constructs unary operation expression for given input expression.
    VectorUnaryOp(const E &u);

    //! Size of the matrix.
    [[nodiscard]] size_t size() const;

    //! Returns vector element at i.
    T operator[](size_t i) const;

private:
    const E &u_;
    Op op_;
};

// MARK: - VectorUnaryOp Aliases

//! Matrix expression for type casting.
template <typename T, typename E, typename U>
using VectorTypeCast = VectorUnaryOp<T, E, TypeCast<U, T>>;

// MARK: - VectorBinaryOp

//!
//! \brief Vector expression for binary operation.
//!
//! This vector expression represents a binary vector operation that takes
//! two input vector expressions.
//!
//! \tparam T - Real number type.
//! \tparam E1 - First input expression type.
//! \tparam E2 - Second input expression type.
//! \tparam Op - Binary operation.
//!
template <typename T, typename E1, typename E2, typename Op>
class VectorBinaryOp : public VectorExpression<T, VectorBinaryOp<T, E1, E2, Op>> {
public:
    //! Constructs binary operation expression for given input vector
    //! expressions.
    VectorBinaryOp(const E1 &u, const E2 &v);

    //! Size of the matrix.
    [[nodiscard]] size_t size() const;

    //! Returns vector element at i.
    T operator[](size_t i) const;

private:
    const E1 &u_;
    const E2 &v_;
    Op op_;
};

//!
//! \brief Vector expression for matrix-scalar binary operation.
//!
//! This vector expression represents a binary matrix operation that takes
//! one input vector expression and one scalar.
//!
//! \tparam T - Real number type.
//! \tparam E - Input expression type.
//! \tparam Op - Binary operation.
//!
template <typename T, typename E, typename Op>
class VectorScalarBinaryOp : public VectorExpression<T, VectorScalarBinaryOp<T, E, Op>> {
public:
    //! Constructs a binary expression for given vector and scalar.
    VectorScalarBinaryOp(const E &u, const T &v);

    //! Size of the matrix.
    [[nodiscard]] size_t size() const;

    //! Returns vector element at i.
    T operator[](size_t i) const;

private:
    const E &u_;
    T v_;
    Op op_;
};

// MARK: - VectorBinaryOp Aliases

//! Vector-vector addition expression.
template <typename T, typename E1, typename E2>
using VectorAdd = VectorBinaryOp<T, E1, E2, std::plus<T>>;

//! Vector-scalar addition expression.
template <typename T, typename E>
using VectorScalarAdd = VectorScalarBinaryOp<T, E, std::plus<T>>;

//! Vector-vector addition expression.
template <typename T, typename E1, typename E2>
using VectorSub = VectorBinaryOp<T, E1, E2, std::minus<T>>;

//! Vector-scalar subtraction expression.
template <typename T, typename E>
using VectorScalarSub = VectorScalarBinaryOp<T, E, std::minus<T>>;

//! Scalar-vector subtraction expression.
template <typename T, typename E>
using VectorScalarRSub = VectorScalarBinaryOp<T, E, RMinus<T>>;

//! Element-wise vector-vector multiplication expression.
template <typename T, typename E1, typename E2>
using VectorMul = VectorBinaryOp<T, E1, E2, std::multiplies<T>>;

//! Vector-scalar multiplication expression.
template <typename T, typename E>
using VectorScalarMul = VectorScalarBinaryOp<T, E, std::multiplies<T>>;

//! Element-wise vector-vector division expression.
template <typename T, typename E1, typename E2>
using VectorDiv = VectorBinaryOp<T, E1, E2, std::divides<T>>;

//! Vector-scalar division expression.
template <typename T, typename E>
using VectorScalarDiv = VectorScalarBinaryOp<T, E, std::divides<T>>;

//! Scalar-vector division expression.
template <typename T, typename E>
using VectorScalarRDiv = VectorScalarBinaryOp<T, E, RDivides<T>>;

// MARK: - Global Functions

//! Scalar-vector addition operation.
template <typename T, typename E>
VectorScalarAdd<T, E> operator+(const T &a, const VectorExpression<T, E> &b);

//! Vector-scalar addition operation.
template <typename T, typename E>
VectorScalarAdd<T, E> operator+(const VectorExpression<T, E> &a, const T &b);

//! Vector-vector addition operation.
template <typename T, typename E1, typename E2>
VectorAdd<T, E1, E2> operator+(const VectorExpression<T, E1> &a, const VectorExpression<T, E2> &b);

//! Scalar-vector subtraction operation.
template <typename T, typename E>
VectorScalarRSub<T, E> operator-(const T &a, const VectorExpression<T, E> &b);

//! Scalar-vector addition operation.
template <typename T, typename E>
VectorScalarSub<T, E> operator-(const VectorExpression<T, E> &a, const T &b);

//! Vector-vector addition operation.
template <typename T, typename E1, typename E2>
VectorSub<T, E1, E2> operator-(const VectorExpression<T, E1> &a, const VectorExpression<T, E2> &b);

//! Scalar-vector multiplication operation.
template <typename T, typename E>
VectorScalarMul<T, E> operator*(const T &a, const VectorExpression<T, E> &b);

//! Vector-scalar multiplication operation.
template <typename T, typename E>
VectorScalarMul<T, E> operator*(const VectorExpression<T, E> &a, const T &b);

//! Element-wise vector-vector multiplication operation.
template <typename T, typename E1, typename E2>
VectorMul<T, E1, E2> operator*(const VectorExpression<T, E1> &a, const VectorExpression<T, E2> &b);

//! Scalar-vector division operation.
template <typename T, typename E>
VectorScalarRDiv<T, E> operator/(const T &a, const VectorExpression<T, E> &b);

//! Vector-scalar division operation.
template <typename T, typename E>
VectorScalarDiv<T, E> operator/(const VectorExpression<T, E> &a, const T &b);

//! Element-wise vector-vector division operation.
template <typename T, typename E1, typename E2>
VectorDiv<T, E1, E2> operator/(const VectorExpression<T, E1> &a, const VectorExpression<T, E2> &b);

}  // namespace vox

#include "math/vector_expression-inl.h"
