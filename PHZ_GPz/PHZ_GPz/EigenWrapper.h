/**
 * @file PHZ_GPz/EigenWrapper.h
 * @date 11/29/18
 * @author user
 *
 * @copyright (C) 2012-2020 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3.0 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

#ifndef _PHZ_GPZ_EIGEN_WRAPPER_H
#define _PHZ_GPZ_EIGEN_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace PHZ_GPz {

    // ==============
    // Shortcut types
    // ==============

    using Mat2d = Eigen::MatrixXd;
    using Mat1d = Eigen::VectorXd;
    using Vec2d = Eigen::ArrayXXd;
    using Vec1d = Eigen::ArrayXd;
    using Vec1i = Eigen::ArrayXi;

    using MapMat2d = Eigen::Map<Eigen::MatrixXd>;
    using MapMat1d = Eigen::Map<Eigen::VectorXd>;
    using MapVec2d = Eigen::Map<Eigen::ArrayXXd>;
    using MapVec1d = Eigen::Map<Eigen::ArrayXd>;

    template<typename MatrixType>
    double computeLogDeterminant(const Eigen::LLT<MatrixType>& cholesky) {
        auto& lower = cholesky.matrixL();

        double logDet = 0.0;
        for (int i = 0; i < lower.rows(); ++i) {
            logDet += log(lower(i,i));
        }

        return 2*logDet;
    }

    template<typename MatrixType>
    double computeLogDeterminant(const Eigen::LDLT<MatrixType>& cholesky) {
        auto& diag = cholesky.vectorD();

        double logDet = 0.0;
        for (int i = 0; i < diag.size(); ++i) {
            logDet += log(diag[i]);
        }

        return logDet;
    }

    template<typename MatrixType>
    double computeLogDeterminant(const Eigen::JacobiSVD<MatrixType>& svd) {
        auto& diag = svd.singularValues();

        double logDet = 0.0;
        for (int i = 0; i < diag.size(); ++i) {
            logDet += log(diag[i]);
        }

        return logDet;
    }

    template<typename MatrixType>
    double computeLogDeterminant(const MatrixType& matrix) {
        Eigen::JacobiSVD<Mat2d> svd(matrix);
        return computeLogDeterminant(svd);
    }

    template<typename MatrixType>
    Mat2d computeInverseSymmetric(const MatrixType& matrix, const Eigen::JacobiSVD<MatrixType>& svd) {
        assert(matrix.rows() == matrix.cols() && "can only be called on symmetric matrices");
        return svd.solve(MatrixType::Identity(matrix.rows(),matrix.rows()));
    }

    template<typename MatrixType>
    Mat2d computeInverseSymmetric(const MatrixType& matrix, const Eigen::LDLT<MatrixType>& ldlt) {
        assert(matrix.rows() == matrix.cols() && "can only be called on symmetric matrices");
        return ldlt.solve(MatrixType::Identity(matrix.rows(),matrix.rows()));
    }

    template<typename MatrixType>
    Mat2d computeInverseSymmetric(const MatrixType& matrix, const Eigen::LLT<MatrixType>& chol) {
        assert(matrix.rows() == matrix.cols() && "can only be called on symmetric matrices");
        return chol.solve(MatrixType::Identity(matrix.rows(),matrix.rows()));
    }

    template<typename MatrixType>
    Mat2d computeInverseSymmetric(const MatrixType& matrix) {
        assert(matrix.rows() == matrix.cols() && "can only be called on symmetric matrices");
        Eigen::JacobiSVD<Mat2d> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        return svd.solve(Mat2d::Identity(matrix.rows(),matrix.rows()));
    }

    namespace Implementation {
        // ====================================
        // Iterator types to use STL algorithms
        // ====================================

        template<typename T>
        class EigenIterator;

        template<typename T>
        class EigenConstIterator {
            const T* data_ = nullptr;
            std::ptrdiff_t index_ = 0;

        public:

            EigenConstIterator(const T& data, std::ptrdiff_t index) : data_(&data), index_(index) {}

            EigenConstIterator() = default;
            EigenConstIterator(const EigenConstIterator&) = default;
            EigenConstIterator(EigenConstIterator&&) = default;

            EigenConstIterator(const EigenIterator<T>& iter) : data_(iter.data_), index_(iter.index_) {}

            EigenConstIterator& operator=(const EigenConstIterator&) = default;
            EigenConstIterator& operator=(EigenConstIterator&&) = default;

            EigenConstIterator operator ++ (int) {
                EigenConstIterator iter = *this;
                ++iter;
                return iter;
            }

            EigenConstIterator& operator ++ () {
                ++index_; return *this;
            }

            EigenConstIterator operator -- (int) {
                EigenConstIterator iter = *this;
                --iter;
                return iter;
            }

            EigenConstIterator& operator -- () {
                --index_; return *this;
            }

            void operator += (int n) {
                index_ += n;
            }

            void operator -= (int n) {
                index_ -= n;
            }

            EigenConstIterator operator + (int n) const {
                EigenConstIterator iter = *this;
                iter += n;
                return iter;
            }

            EigenConstIterator operator - (int n) const {
                EigenConstIterator iter = *this;
                iter -= n;
                return iter;
            }

            std::ptrdiff_t operator - (const EigenConstIterator& iter) const {
                return index_ - iter.index_;
            }

            bool operator == (const EigenConstIterator& iter) const {
                return index_ == iter.index_;
            }

            bool operator != (const EigenConstIterator& iter) const {
                return index_ != iter.index_;
            }

            bool operator < (const EigenConstIterator& iter) const {
                return index_ < iter.index_;
            }

            bool operator <= (const EigenConstIterator& iter) const {
                return index_ <= iter.index_;
            }

            bool operator > (const EigenConstIterator& iter) const {
                return index_ > iter.index_;
            }

            bool operator >= (const EigenConstIterator& iter) const {
                return index_ >= iter.index_;
            }

            auto operator * () -> decltype((*data_)[index_]) {
                return (*data_)[index_];
            }

            auto operator -> () -> decltype(&(*data_)[index_]) {
                return &(*data_)(index_);
            }
        };

        template<typename T>
        class EigenIterator {
            T* data_ = nullptr;
            std::ptrdiff_t index_ = 0;

            friend class EigenConstIterator<T>;

        public:

            EigenIterator(T& data, std::ptrdiff_t index) : data_(&data), index_(index) {}

            EigenIterator() = default;
            EigenIterator(const EigenIterator&) = default;
            EigenIterator(EigenIterator&&) = default;

            EigenIterator& operator=(const EigenIterator&) = default;
            EigenIterator& operator=(EigenIterator&&) = default;

            EigenIterator operator ++ (int) {
                EigenIterator iter = *this;
                ++iter;
                return iter;
            }

            EigenIterator& operator ++ () {
                ++index_; return *this;
            }

            EigenIterator operator -- (int) {
                EigenIterator iter = *this;
                --iter;
                return iter;
            }

            EigenIterator& operator -- () {
                --index_; return *this;
            }

            void operator += (int n) {
                index_ += n;
            }

            void operator -= (int n) {
                index_ -= n;
            }

            EigenIterator operator + (int n) const {
                EigenIterator iter = *this;
                iter += n;
                return iter;
            }

            EigenIterator operator - (int n) const {
                EigenIterator iter = *this;
                iter -= n;
                return iter;
            }

            std::ptrdiff_t operator - (const EigenIterator& iter) const {
                return index_ - iter.index_;
            }

            bool operator == (const EigenIterator& iter) const {
                return index_ == iter.index_;
            }

            bool operator != (const EigenIterator& iter) const {
                return index_ != iter.index_;
            }

            bool operator < (const EigenIterator& iter) const {
                return index_ < iter.index_;
            }

            bool operator <= (const EigenIterator& iter) const {
                return index_ <= iter.index_;
            }

            bool operator > (const EigenIterator& iter) const {
                return index_ > iter.index_;
            }

            bool operator >= (const EigenIterator& iter) const {
                return index_ >= iter.index_;
            }

            auto operator * () -> decltype((*data_)[index_]) {
                return (*data_)[index_];
            }

            auto operator -> () -> decltype(&(*data_)[index_]) {
                return &(*data_)(index_);
            }
        };
    }  // namespace Implementation

}  // namespace PHZ_GPz

namespace Eigen {

    // =========================================
    // Specialization of std::begin and std::end
    // =========================================

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenConstIterator<Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        begin(const Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenConstIterator<DataType>(mat, 0);
    }

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenConstIterator<Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        end(const Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenConstIterator<DataType>(mat, mat.size());
    }

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenConstIterator<Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        begin(const Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenConstIterator<DataType>(mat, 0);
    }

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenConstIterator<Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        end(const Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenConstIterator<DataType>(mat, mat.size());
    }

    template<typename Matrix>
    PHZ_GPz::Implementation::EigenConstIterator<Eigen::Map<Matrix>>
        begin(const Eigen::Map<Matrix>& mat) {

        using DataType = Eigen::Map<Matrix>;
        return PHZ_GPz::Implementation::EigenConstIterator<DataType>(mat, 0);
    }

    template<typename Matrix>
    PHZ_GPz::Implementation::EigenConstIterator<Eigen::Map<Matrix>>
        end(const Eigen::Map<Matrix>& mat) {

        using DataType = Eigen::Map<Matrix>;
        return PHZ_GPz::Implementation::EigenConstIterator<DataType>(mat, mat.size());
    }


    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenIterator<Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        begin(Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenIterator<DataType>(mat, 0);
    }

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenIterator<Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        end(Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Matrix<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenIterator<DataType>(mat, mat.size());
    }

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenIterator<Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        begin(Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenIterator<DataType>(mat, 0);
    }

    template<typename Scalar, int RowsAtCompileType, int ColsAtCompileTime>
    PHZ_GPz::Implementation::EigenIterator<Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>>
        end(Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>& mat) {

        using DataType = Eigen::Array<Scalar,RowsAtCompileType,ColsAtCompileTime>;
        return PHZ_GPz::Implementation::EigenIterator<DataType>(mat, mat.size());
    }

    template<typename Matrix>
    PHZ_GPz::Implementation::EigenIterator<Eigen::Map<Matrix>>
        begin(Eigen::Map<Matrix>& mat) {

        using DataType = Eigen::Map<Matrix>;
        return PHZ_GPz::Implementation::EigenIterator<DataType>(mat, 0);
    }

    template<typename Matrix>
    PHZ_GPz::Implementation::EigenIterator<Eigen::Map<Matrix>>
        end(Eigen::Map<Matrix>& mat) {

        using DataType = Eigen::Map<Matrix>;
        return PHZ_GPz::Implementation::EigenIterator<DataType>(mat, mat.size());
    }

}  // namespace Eigen

#endif
