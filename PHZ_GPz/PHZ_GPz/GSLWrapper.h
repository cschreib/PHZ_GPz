/**
 * @file PHZ_GPz/GSLWrapper.h
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

#ifndef _PHZ_GPZ_GSL_WRAPPER_H
#define _PHZ_GPZ_GSL_WRAPPER_H

#include "PHZ_GPz/EigenWrapper.h"
#include <limits>
#include <gsl/gsl_multimin.h>

namespace PHZ_GPz {
namespace Minimize {

    struct Options {
        double initialStep = 0.1;
        double relativeTolerance = 1e-3;
        uint_t maxIterations = 1000;
        bool   hasValidation = false;
        uint_t maxValidationAttempts = 5;
    };

    struct Result {
        bool success = false;
        Vec1d parameters;
        double metric = std::numeric_limits<double>::quiet_NaN();
        uint_t numberIterations = 0;
    };

    enum class FunctionOutput {
        METRIC_TRAIN, METRIC_VALID, DERIVATIVES, ALL
    };

    template<typename F>
    Result minimizeBFGS(const Options& options, const Vec1d& initial, F&& function) {
        using functionPointer = typename std::decay<decltype(function)>::type*;

        // Initialize return value
        Result result;
        result.parameters.resize(initial.size());

        const uint_t n = initial.size();

        // Initialize minimized function
        gsl_multimin_function_fdf mf;
        mf.n = n;
        mf.params = reinterpret_cast<void*>(&function);

        mf.f = [](const gsl_vector* xGSL, void* data) {
            const uint_t tn = xGSL->size;
            Vec1d xEigen(tn);
            for (uint_t i = 0; i < tn; ++i) {
                xEigen[i] = gsl_vector_get(xGSL, i);
            }

            functionPointer fp = reinterpret_cast<functionPointer>(data);
            Vec1d output = (*fp)(xEigen, FunctionOutput::METRIC_TRAIN);

            return output[0];
        };

        mf.df = [](const gsl_vector* xGSL, void* data, gsl_vector* derivGSL) {
            const uint_t tn = xGSL->size;
            Vec1d xEigen(tn);
            for (uint_t i = 0; i < tn; ++i) {
                xEigen[i] = gsl_vector_get(xGSL, i);
            }

            functionPointer fp = reinterpret_cast<functionPointer>(data);
            Vec1d output = (*fp)(xEigen, FunctionOutput::DERIVATIVES);

            for (uint_t i = 0; i < tn; ++i) {
                gsl_vector_set(derivGSL, i, output[i+1]);
            }
        };

        mf.fdf = [](const gsl_vector* xGSL, void* data, double* metric, gsl_vector* derivGSL) {
            const uint_t tn = xGSL->size;
            Vec1d xEigen(tn);
            for (uint_t i = 0; i < tn; ++i) {
                xEigen[i] = gsl_vector_get(xGSL, i);
            }

            functionPointer fp = reinterpret_cast<functionPointer>(data);
            Vec1d output = (*fp)(xEigen, FunctionOutput::ALL);

            *metric = output[0];

            for (uint_t i = 0; i < tn; ++i) {
                gsl_vector_set(derivGSL, i, output[i+1]);
            }
        };

        gsl_vector* x = nullptr;
        gsl_multimin_fdfminimizer* m = nullptr;
        double bestValid = std::numeric_limits<double>::infinity();
        uint_t noValidationImprovementAttempts = 0;

        try {
            // Initial conditions
            x = gsl_vector_alloc(n);
            for (uint_t i = 0; i < n; ++i) {
                gsl_vector_set(x, i, initial[i]);
            }

            Vec1d xEigen(n);

            // Initialize minimizer
            const gsl_multimin_fdfminimizer_type* T = gsl_multimin_fdfminimizer_vector_bfgs2;
            m = gsl_multimin_fdfminimizer_alloc(T, n);
            gsl_multimin_fdfminimizer_set(m, &mf, x, options.initialStep, options.relativeTolerance);

            // Minimization loop
            int status;

            do {
                ++result.numberIterations;

                status = gsl_multimin_fdfminimizer_iterate(m);
                if (status != 0) {
                    if (status == GSL_ENOPROG) {
                        // No progress possible, local minimum found
                        result.success = true;
                    }

                    break;
                }

                status = gsl_multimin_test_gradient(m->gradient, options.relativeTolerance);
                if (status == GSL_SUCCESS) {
                    // Gradient small enough, local minimum found
                    result.success = true;
                    break;
                }

                for (uint_t i = 0; i < n; ++i) {
                    xEigen[i] = gsl_vector_get(m->x, i);
                }

                if (options.hasValidation) {
                    Vec1d valid = function(xEigen, FunctionOutput::METRIC_VALID);
                    if (valid[0] < bestValid) {
                        bestValid = valid[0];
                        noValidationImprovementAttempts = 0;
                    } else {
                        ++noValidationImprovementAttempts;
                    }

                    if (noValidationImprovementAttempts == options.maxValidationAttempts) {
                        result.success = true;
                        break;
                    }
                }
            } while (status == GSL_CONTINUE && result.numberIterations < options.maxIterations);

            // Write output
            for (uint_t i = 0; i < n; ++i) {
                result.parameters[i] = gsl_vector_get(m->x, i);
            }

            result.metric = m->f;

            // Cleanup
            gsl_multimin_fdfminimizer_free(m);
            gsl_vector_free(x);
        } catch (...) {
            // Cleanup
            if (m) gsl_multimin_fdfminimizer_free(m);
            if (x) gsl_vector_free(x);
        }

        return result;
    }

}
}

#endif
