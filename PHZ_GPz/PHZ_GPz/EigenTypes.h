/**
 * @file PHZ_GPz/EigenTypes.h
 * @date 05/29/19
 * @author cschreib
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

#ifndef _PHZ_GPZ_EIGEN_TYPES_H
#define _PHZ_GPZ_EIGEN_TYPES_H

#include <Eigen/Dense>

namespace PHZ_GPz {

    // ==============
    // Shortcut types
    // ==============

    using Mat2d = Eigen::MatrixXd;
    using Mat1d = Eigen::VectorXd;
    using Vec2d = Eigen::ArrayXXd;
    using Vec1d = Eigen::ArrayXd;
    using Vec1i = Eigen::ArrayXi;

}

#endif
