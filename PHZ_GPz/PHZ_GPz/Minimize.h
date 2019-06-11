/**
 * @file PHZ_GPz/Minimize.h
 * @date 05/21/19
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

#ifndef _PHZ_GPZ_MINIMIZE_H
#define _PHZ_GPZ_MINIMIZE_H

#include "PHZ_GPz/Utils.h"
#include "PHZ_GPz/EigenTypes.h"

#include <limits>

namespace PHZ_GPz {
namespace Minimize {

    struct Options {
        double initialStep = 0.1;
        double minimizerTolerance = 1e-3;
        double gradientTolerance = 1e-3;
        uint_t maxIterations = 1000;
        bool   hasValidation = false;
        uint_t maxValidationAttempts = 50;
        bool   verbose = false;
        bool   verboseSingleLine = false;
    };

    struct Result {
        bool   success = false;
        Vec1d  parameters;
        Vec1d  parametersBestValid;
        double metric = std::numeric_limits<double>::quiet_NaN();
        double metricValid = std::numeric_limits<double>::quiet_NaN();
        double metricBestValid = std::numeric_limits<double>::quiet_NaN();
        double metricValidBestValid = std::numeric_limits<double>::quiet_NaN();
        uint_t numberIterations = 0;
        uint_t numberIterationsBestValid = 0;
    };

    enum class FunctionOutput {
        METRIC_TRAIN, METRIC_VALID, DERIVATIVES_TRAIN, ALL_TRAIN
    };

}  // namespace Minimize
}  // namespace PHZ_GPz

#endif
