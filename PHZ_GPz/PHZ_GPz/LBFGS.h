/**
 * @file PHZ_GPz/LBFGS.h
 * @date 02/12/19
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

#ifndef _PHZ_GPZ_LBFGS_H
#define _PHZ_GPZ_LBFGS_H

#include "PHZ_GPz/Utils.h"
#include "PHZ_GPz/EigenTypes.h"
#include "PHZ_GPz/Minimize.h"

#include <iostream>

namespace PHZ_GPz {
namespace Minimize {

    namespace ImplementationLBFGS {
        bool checkLegalMove(double metric) {
            if (std::isnan(metric) || !std::isfinite(metric)) {
                return false;
            }

            return true;
        }

        bool checkLegalMove(const Vec1d& gradient) {
            for (double g : gradient) {
                if (!checkLegalMove(g)) {
                    return false;
                }
            }

            return true;
        }

        bool checkLegalMove(double metric, const Vec1d& gradient) {
            return checkLegalMove(metric) && checkLegalMove(gradient);
        }

        double minPolynomial3(double x0, double y0, double dy0, double x1, double y1, double dy1,
            double xmin, double xmax) {

            // Find the location of minimum value of function interpolated with a third order
            // polynomial using points (x0,y0) and (x1,y1) with known derivatives.

            double xlow, xup, ylow, yup, dylow, dyup;
            if (x0 < x1) {
                xlow = x0; ylow = y0; dylow = dy0;
                xup  = x1; yup  = y1; dyup  = dy1;
            } else {
                xlow = x1; ylow = y1; dylow = dy1;
                xup  = x0; yup  = y0; dyup  = dy0;
            }

            double d1 = dylow + dyup - 3.0*(ylow - yup)/(xlow - xup);
            double d2 = d1*d1 - dyup*dylow;
            if (d2 >= 0.0) {
                d2 = sqrt(d2);
                double xtemp = xup - (xup - xlow)*((dyup + d2 - d1)/(dyup - dylow + 2.0*d2));
                if (xtemp < xmin) {
                    xtemp = xmin;
                }
                if (xtemp > xmax) {
                    xtemp = xmax;
                }

                return xtemp;
            } else {
                return 0.5*(xmin + xmax);
            }
        }

        double minPolynomial2(double x0, double y0, double dy0, double x1, double y1,
            double xmin, double xmax) {

            // Find the location of minimum value of function interpolated with a
            // second order polynomial using points (x0,y0) and (x1,y1) with known derivative
            // only for first point.

            // Find best polynomial coefficients
            // y = p0*x*x + p1*x + p2
            double p[3];
            p[0] = ((y1 - y0) - dy0*(x1 - x0))/((x1 - x0)*(x1 - x0));
            p[1] = dy0 - 2.0*p[0]*x0;
            p[2] = y0 + p[0]*x0*x0 - dy0*x0;

            // Test critical points
            double cp[5] = {xmin, xmax, x0, x1, -0.25*p[1]/p[0]};
            double ymin = std::numeric_limits<double>::infinity();
            double xtemp = 0.5*(xmax + xmin);
            for (uint_t i = 0; i < 5; ++i) {
                if (std::isfinite(cp[i]) && cp[i] >= xmin && cp[i] <= xmax) {
                    double ytemp = p[0]*cp[i]*cp[i] + p[1]*cp[i] + p[0];
                    if (std::isfinite(ytemp) && ytemp < ymin) {
                        ymin = ytemp;
                        xtemp = cp[i];
                    }
                }
            }

            return xtemp;
        }

        template<typename F>
        void armijoLineSearch(const Options& options, const Vec1d& x, double& gradientStep,
            const Vec1d& direction, double& metric, Vec1d& gradient, double oldGdt, F&& function) {

            double oldMetric = metric;
            Vec1d oldGradient = gradient;

            Vec1d output = function(x + gradientStep*direction, FunctionOutput::ALL_TRAIN);
            metric = output[0];
            gradient = output.tail(x.size());

            const double c1 = 1e-4;

            while (metric > oldMetric + c1*gradientStep*oldGdt || !checkLegalMove(metric)) {
                double gradientStepNew;
                if (!checkLegalMove(metric)) {
                    // Ignore value of new point
                    gradientStepNew = 0.5*gradientStep;
                } else if (!checkLegalMove(gradient)) {
                    // Use function value of new point but not its derivative
                    // Quadratic interpolation
                    gradientStepNew = ImplementationLBFGS::minPolynomial2(
                        0.0,          oldMetric, oldGdt,
                        gradientStep, metric,
                        0.0, gradientStep
                    );
                } else {
                    // Use function value and derivative of new point
                    // Cubic interpolation
                    gradientStepNew = ImplementationLBFGS::minPolynomial3(
                        0.0,          oldMetric, oldGdt,
                        gradientStep, metric, (gradient*direction).sum(),
                        0.0, gradientStep
                    );
                }

                // Adjust if change in step is too small/large
                if (gradientStepNew < gradientStep*1e-3) {
                    gradientStepNew = gradientStep*1e-3;
                } else if (gradientStepNew > gradientStep*0.6) {
                    gradientStepNew = gradientStep*0.6;
                }

                gradientStep = gradientStepNew;

                // Evaluate new point
                output = function(x + gradientStep*direction, FunctionOutput::ALL_TRAIN);
                metric = output[0];
                gradient = output.tail(x.size());

                // Check whether step size has become too small
                if ((gradientStep*direction).abs().maxCoeff() <= options.minimizerTolerance) {
                    gradientStep = 0.0;
                    metric = oldMetric;
                    gradient = oldGradient;
                    break;
                }
            }
        }


        template<typename F>
        void wolfeLineSearch(const Options& options, const Vec1d& x, double& gradientStep,
            const Vec1d& direction, double& metric, Vec1d& gradient, double gdt, F&& function) {

            double oldMetric = metric;
            Vec1d oldGradient = gradient;
            double oldGdt = gdt;

            Vec1d output = function(x + gradientStep*direction, FunctionOutput::ALL_TRAIN);
            metric = output[0];
            gradient = output.tail(x.size());
            gdt = (gradient*direction).sum();

            double gdtPrev = oldGdt;
            double gradientStepPrev = 0.0;
            double metricPrev = oldMetric;
            Vec1d gradientPrev = oldGradient;
            double normD = direction.abs().maxCoeff();
            uint_t iter = 0;
            uint_t maxIter = 25;

            const double c1 = 1e-4;
            const double c2 = 0.9;

            double bracketStep[2] = {0,0};
            double bracketMetric[2] = {0,0};
            Vec1d bracketGradient[2];

            // Bracketing phase
            // ----------------

            bool done = false;

            while (iter < maxIter) {
                if (!checkLegalMove(metric, gradient)) {
                    // Illegal move, fall back to Armijo line search
                    gradientStep = 0.5*(gradientStep + gradientStepPrev);
                    metric = oldMetric;
                    gradient = oldGradient;
                    gdt = oldGdt;
                    ImplementationLBFGS::armijoLineSearch(options, x, gradientStep, direction,
                        metric, gradient, gdt, function);
                    return;
                }

                if (metric > oldMetric + c1*gradientStep*oldGdt ||
                    (iter > 1 && metric > metricPrev)) {
                    bracketStep[0] = gradientStepPrev;
                    bracketStep[1] = gradientStep;
                    bracketMetric[0] = metricPrev;
                    bracketMetric[1] = metric;
                    bracketGradient[0] = gradientPrev;
                    bracketGradient[1] = gradient;
                    break;
                } else if (std::abs(gdt) <= -c2*oldGdt) {
                    bracketStep[0] = gradientStep;
                    bracketStep[1] = gradientStep;
                    bracketMetric[0] = metric;
                    bracketMetric[1] = metric;
                    bracketGradient[0] = gradient;
                    bracketGradient[1] = gradient;
                    done = true;
                    break;
                } else if (gdt >= 0.0) {
                    bracketStep[0] = gradientStepPrev;
                    bracketStep[1] = gradientStep;
                    bracketMetric[0] = metricPrev;
                    bracketMetric[1] = metric;
                    bracketGradient[0] = gradientPrev;
                    bracketGradient[1] = gradient;
                    break;
                }

                double minStep = gradientStep + 0.01*(gradientStep - gradientStepPrev);
                double maxStep = gradientStep*10.0;

                double gradientStepNew = ImplementationLBFGS::minPolynomial3(
                    gradientStepPrev, metricPrev, gdtPrev,
                    gradientStep,     metric,     gdt,
                    minStep, maxStep
                );

                gradientStepPrev = gradientStep;
                gradientStep = gradientStepNew;

                metricPrev = metric;
                gradientPrev = gradient;
                gdtPrev = gdt;

                output = function(x + gradientStep*direction, FunctionOutput::ALL_TRAIN);
                metric = output[0];
                gradient = output.tail(x.size());
                gdt = (gradient*direction).sum();

                ++iter;
            }

            if (iter == maxIter) {
                bracketStep[0] = 0.0;
                bracketStep[1] = gradientStep;
                bracketMetric[0] = metricPrev;
                bracketMetric[1] = metric;
                bracketGradient[0] = gradientPrev;
                bracketGradient[1] = gradient;
            }

            // Zoom phase
            // ----------

            // We now either have a point satisfying the criteria, or a bracket
            // surrounding a point satisfying the criteria.
            // Refine the bracket until we find a point satisfying the criteria.

            bool insufficientProgress = false;

            while (!done && iter < maxIter) {
                // Find high and low points in bracket
                uint_t ilow, iup;
                if (bracketMetric[0] < bracketMetric[1]) {
                    ilow = 0;
                    iup = 1;
                } else {
                    ilow = 1;
                    iup = 0;
                }

                // Compute new trial value
                if (!checkLegalMove(bracketMetric[0], bracketGradient[0]) ||
                    !checkLegalMove(bracketMetric[1], bracketGradient[1])) {
                    // Bisecting
                    gradientStep = 0.5*(bracketStep[0] + bracketStep[1]);
                } else {
                    // Grad-cubic interpolation
                    gradientStep = ImplementationLBFGS::minPolynomial3(
                        bracketStep[0], bracketMetric[0], (bracketGradient[0]*direction).sum(),
                        bracketStep[1], bracketMetric[1], (bracketGradient[1]*direction).sum(),
                        bracketStep[0], bracketStep[1]
                    );
                }

                // Test that we are making sufficient progress
                double maxStep, minStep;
                if (bracketStep[0] < bracketStep[1]) {
                    minStep = bracketStep[0];
                    maxStep = bracketStep[1];
                } else {
                    minStep = bracketStep[1];
                    maxStep = bracketStep[0];
                }

                if (std::min(maxStep - gradientStep, gradientStep - minStep)/(maxStep - minStep) < 0.1) {
                    // Interpolation close to boundary
                    if (insufficientProgress || gradientStep >= maxStep || gradientStep <= minStep) {
                        // Evaluating at 0.1 from boundary
                        if (std::abs(gradientStep - maxStep) < std::abs(gradientStep - minStep)) {
                            gradientStep = maxStep - 0.1*(maxStep - minStep);
                        } else {
                            gradientStep = minStep + 0.1*(maxStep - minStep);
                        }

                        insufficientProgress = false;
                    } else {
                        insufficientProgress = true;
                    }
                } else {
                    insufficientProgress = false;
                }

                // Evaluate new point
                output = function(x + gradientStep*direction, FunctionOutput::ALL_TRAIN);
                metric = output[0];
                gradient = output.tail(x.size());
                gdt = (gradient*direction).sum();

                bool armijoCondition = metric < oldMetric + c1*gradientStep*oldGdt;
                if (!armijoCondition || metric > bracketMetric[ilow]) {
                    // Armijo condition not satisfied or not lower than lowest bracket point
                    bracketStep[iup] = gradientStep;
                    bracketMetric[iup] = metric;
                    bracketGradient[iup] = gradient;
                } else {
                    if (std::abs(gdt) < -c2*oldGdt) {
                        // Wolfe condition satisfied
                        done = true;
                    } else if (gdt*(bracketStep[iup] - bracketStep[ilow]) >= 0.0) {
                        // Old low becomes new high
                        bracketStep[iup] = bracketStep[ilow];
                        bracketMetric[iup] = bracketMetric[ilow];
                        bracketGradient[iup] = bracketGradient[ilow];
                    }

                    // New point becomes new low
                    bracketStep[ilow] = gradientStep;
                    bracketMetric[ilow] = metric;
                    bracketGradient[ilow] = gradient;
                }

                if (!done && std::abs(bracketStep[1] - bracketStep[0])*normD < options.minimizerTolerance) {
                    // Line search below tolerance
                    break;
                }
            }

            // Find low point in bracket
            uint_t ilow = (bracketMetric[0] < bracketMetric[1] ? 0 : 1);
            gradientStep = bracketStep[ilow];
            metric = bracketMetric[ilow];
            gradient = bracketGradient[ilow];
        }
    }

    template<typename F>
    Result minimizeLBFGS(const Options& options, const Vec1d& initial, F&& function) {
        // Initialize return value
        Result result;
        result.parameters.resize(initial.size());
        if (options.hasValidation) {
            result.parametersBestValid.resize(initial.size());
        }

        const uint_t n = initial.size();

        // Setup validation
        if (options.hasValidation) {
            result.metricValidBestValid = std::numeric_limits<double>::infinity();
        }

        double currentValid = 0.0;
        uint_t noValidationImprovementAttempts = 0;

        if (options.verbose && options.verboseSingleLine) {
            std::cout << "iter: 0, metric train: --";
            if (options.hasValidation) {
                std::cout << ", metric valid: -- (best: --)";
            }
            std::cout << "\r" << std::flush;
        }

        // Fixed configurations
        const uint_t c = 100;

        // Setup iterations
        Vec1d x = initial;
        Vec1d output = function(x, FunctionOutput::ALL_TRAIN);
        double metric = output[0];
        double oldMetric = metric;
        Vec1d gradient = output.tail(n);
        Vec1d previousGradient = gradient;
        Vec1d direction = gradient;
        Vec2d S(n,c); S.fill(0.0);
        Vec2d Y(n,c); Y.fill(0.0);
        Vec1d Ys(c);  Ys.fill(0.0);
        double Hdiag = 1.0;
        double gradientStep = 1.0;

        // LBFGS correction cache: [lbfgsStart - lbfgsEnd]
        // lbfgsStart: first value to use in the cache (inclusive)
        // lbfgsEnd: first value to not use in the cache (exclusive)
        uint_t lbfgsStart = 0;
        uint_t lbfgsEnd = 0;

        // Early exit for already optimal conditions
        double optimalConditions = gradient.abs().maxCoeff();
        if (optimalConditions < options.gradientTolerance) {
            result.parameters = x;
            result.metric = metric;
            result.success = true;
            return result;
        }

        // Minimization loop
        // -----------------

        std::string convergence_criterion;

        for (result.numberIterations = 1;
            result.numberIterations < options.maxIterations && options.maxIterations != 0;
            ++result.numberIterations) {

            // Compute direction
            // -----------------

            if (result.numberIterations == 1) {
                direction = -gradient;
            } else {
                Vec1d changeGradient = gradient - previousGradient;
                Vec1d movement = gradientStep*direction;

                double ys = (movement*changeGradient).sum();
                if (ys > 1e-10) {
                    ++lbfgsEnd;
                    if (lbfgsEnd > c) {
                        lbfgsEnd = 0;
                        lbfgsStart = 1;
                    } else if (lbfgsStart != 0) {
                        ++lbfgsStart;
                        if (lbfgsStart == c) {
                            lbfgsStart = 0;
                        }
                    }

                    uint_t lbfgsBack = lbfgsEnd;
                    if (lbfgsBack == 0) {
                        lbfgsBack = c-1;
                    } else {
                        --lbfgsBack;
                    }

                    S.col(lbfgsBack) = movement;
                    Y.col(lbfgsBack) = changeGradient;
                    Ys[lbfgsBack] = ys;

                    Hdiag = ys/(changeGradient*changeGradient).sum();
                }

                uint_t nc = 0;
                if (lbfgsStart == 0) {
                    nc = lbfgsEnd;
                } else {
                    nc = c;
                }

                Vec1d alpha(nc);
                direction = -gradient;
                for (uint_t i = 0, lbfgsBack = lbfgsEnd; i < nc; ++i) {
                    if (lbfgsBack == 0) {
                        lbfgsBack = c-1;
                    } else {
                        --lbfgsBack;
                    }

                    alpha[lbfgsBack] = (S.col(lbfgsBack)*direction).sum()/Ys[lbfgsBack];
                    direction -= alpha[lbfgsBack]*Y.col(lbfgsBack);
                }

                direction *= Hdiag;

                for (uint_t i = 0, lbfgsFront = lbfgsStart; i < nc; ++i) {
                    double beta = (Y.col(lbfgsFront)*direction).sum()/Ys[lbfgsFront];
                    direction += (alpha[lbfgsFront] - beta)*S.col(lbfgsFront);

                    ++lbfgsFront;
                    if (lbfgsFront == c) {
                        lbfgsFront = 0;
                    }
                }
            }

            previousGradient = gradient;

            double gdt = (gradient*direction).sum();

            if (-gdt < options.minimizerTolerance) {
                // Directional derivative below threshold
                convergence_criterion = "directional derivative below threshold";
                result.success = true;
                break;
            }

            // Line search
            // -----------

            // Select initial guess
            if (result.numberIterations == 1) {
                gradientStep = std::min(1.0, 1.0/gradient.abs().sum());
            } else {
                gradientStep = 1.0;
            }

            oldMetric = metric;
            ImplementationLBFGS::wolfeLineSearch(options, x, gradientStep, direction,
                metric, gradient, gdt, function);
            x += gradientStep*direction;

            // Exit conditions
            // ---------------

            // Compute optimality condition
            optimalConditions = gradient.abs().maxCoeff();
            if (optimalConditions < options.gradientTolerance) {
                // Gradient below threshold
                convergence_criterion = "gradient below threshold";
                result.success = true;
                break;
            }

            optimalConditions = (gradientStep*direction).abs().maxCoeff();
            if (optimalConditions < options.minimizerTolerance) {
                // Step size below threshold
                convergence_criterion = "step size below threshold";
                result.success = true;
                break;
            }

            optimalConditions = std::abs(metric - oldMetric);
            if (optimalConditions < options.minimizerTolerance) {
                // Metric change below threshold
                convergence_criterion = "metric delta below threshold";
                result.success = true;
                break;
            }

            if (options.hasValidation) {
                currentValid = function(x, FunctionOutput::METRIC_VALID)[0];
                if (currentValid < result.metricValidBestValid) {
                    result.metricBestValid = metric;
                    result.metricValidBestValid = currentValid;
                    result.numberIterationsBestValid = result.numberIterations;
                    noValidationImprovementAttempts = 0;
                    result.parametersBestValid = x;
                } else {
                    ++noValidationImprovementAttempts;
                }

                if (noValidationImprovementAttempts == options.maxValidationAttempts) {
                    // Validation metric is not improving
                    convergence_criterion = "no improvement in validation metric";
                    result.success = true;
                    break;
                }
            }

            if (options.verbose) {
                std::cout << "iter: " << result.numberIterations
                    << ", metric train: " << metric;
                if (options.hasValidation) {
                    std::cout << ", metric valid: " << currentValid
                        << " (best: " << result.metricValidBestValid << ")";
                }

                if (options.verboseSingleLine) {
                    std::cout << "\r" << std::flush;
                } else {
                    std::cout << std::endl;
                }
            }
        }

        if (options.verbose) {
            std::cout << "iter: " << result.numberIterations
                << ", metric train: " << metric;
            if (options.hasValidation) {
                std::cout << ", metric valid: " << currentValid
                    << " (best: " << result.metricValidBestValid << ")";
            }
            std::cout << std::endl;
            if (result.success) {
                std::cout << "fit converged (" << convergence_criterion << ")" << std::endl;
            } else {
                std::cout << "fit failed to converge (maximum number of iterations reached)" << std::endl;
            }
        }

        // Write output
        result.parameters = x;
        result.metric = metric;
        if (options.hasValidation) {
            result.metricValid = currentValid;
        }

        return result;
    }

}  // namespace Minimize
}  // namespace PHZ_GPz

#endif
