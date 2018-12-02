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
#include <memory>

namespace PHZ_GPz {

    using Mat2d = Eigen::MatrixXd;
    using Mat1d = Eigen::VectorXd;
    using Vec2d = Eigen::ArrayXXd;
    using Vec1d = Eigen::ArrayXd;

    using MapMat2d = Eigen::Map<Eigen::MatrixXd>;
    using MapMat1d = Eigen::Map<Eigen::VectorXd>;
    using MapVec2d = Eigen::Map<Eigen::ArrayXXd>;
    using MapVec1d = Eigen::Map<Eigen::ArrayXd>;

    using uint_t = std::size_t;

}  // namespace PHZ_GPz


#endif
