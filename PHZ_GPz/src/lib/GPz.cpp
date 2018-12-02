/**
 * @file src/lib/GPz.cpp
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

#include "PHZ_GPz/GPz.h"
#include "PHZ_GPz/GSLWrapper.h"

namespace PHZ_GPz {

// =========================
// Internal functions: setup
// =========================

void GPz::updateNumberParameters_() {
    // Pseudo input position
    indexPseudoPosition_ = 0;
    numberParameters_ += numberPseudoInputs_*numberFeatures_;

    // Pseudo input relevance
    indexPseudoRelevance_ = numberParameters_;
    numberParameters_ += numberPseudoInputs_;

    // Pseudo input covariance
    indexPseudoCovariance_ = numberParameters_;
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            numberParameters_ += 1;
            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            numberParameters_ += numberPseudoInputs_;
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            numberParameters_ += numberFeatures_;
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            numberParameters_ += numberFeatures_*numberPseudoInputs_;
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            numberParameters_ += numberFeatures_*(numberFeatures_ + 1)/2;
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            numberParameters_ += numberFeatures_*(numberFeatures_ + 1)/2*numberPseudoInputs_;
            break;
        }
    }

    // Output error parametrization
    indexError_ = numberParameters_;
    switch (outputErrorType_) {
        case OutputErrorType::CONSTANT: {
            numberParameters_ += 1;
            break;
        }
        case OutputErrorType::HETEROSCEDASTIC: {
            // Constant
            numberParameters_ += 1;

            // Weights
            indexErrorWeight_ = numberParameters_;
            numberParameters_ += numberPseudoInputs_;

            // Relevance
            indexErrorRelevance_ = numberParameters_;
            numberParameters_ += numberPseudoInputs_;
            break;
        }
    }

    // Prior mean function
    indexPriorMean_ = numberParameters_;
    switch (priorMean_) {
        case PriorMeanFunction::ZERO: {
            break;
        }
        case PriorMeanFunction::LINEAR: {
            numberParameters_ += numberFeatures_ + 1;
            break;
        }
    };
}

Vec1d GPz::makeParameterArray_() const {
    Vec1d parameters(numberParameters_);

    const uint_t m = numberPseudoInputs_;
    const uint_t d = numberFeatures_;

    // Pseudo input position
    uint_t ip = 0;
    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j < d; ++j) {
        parameters[ip] = pseudoInputPositions_(i,j);
        ++ip;
    }

    // Pseudo input relevances
    for (uint_t i = 0; i < m; ++i) {
        parameters[ip] = pseudoInputRelevances_[i];
        ++ip
    }

    // Pseudo input covariances
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            parameters_[ip] = pseudoInputCovariances_[0](0,0);
            ++ip;
            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            for (uint_t i = 0; i < m; ++i) {
                parameters_[ip] = pseudoInputCovariances_[i](0,0);
                ++ip;
            }
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            for (uint_t j = 0; j < d; ++j) {
                parameters[ip] = pseudoInputCovariances_[0](j,j);
                ++ip;
            }
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j) {
                parameters[ip] = pseudoInputCovariances_[i](j,j);
                ++ip;
            }
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                parameters[ip] = pseudoInputCovariances_[0](j,k);
                ++ip;
            }
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                parameters[ip] = pseudoInputCovariances_[i](j,k);
                ++ip;
            }
            break;
        }
    }

    // Output error parametrization
    switch (outputErrorType_) {
        case OutputErrorType::CONSTANT: {
            parameters[ip] = errorConstant_;
            ++ip;
            break;
        }
        case OutputErrorType::HETEROSCEDASTIC: {
            // Constant
            parameters[ip] = errorConstant_;
            ++ip;

            // Weights
            for (uint_t i = 0; i < m; ++i) {
                parameters[ip] = errorWeights_[i];
                ++ip;
            }

            // Relevance
            for (uint_t i = 0; i < m; ++i) {
                parameters[ip] = errorRelevances_[i];
                ++ip;
            }

            break;
        }
    }

    // Prior mean function
    switch (priorMean_) {
        case PriorMeanFunction::ZERO: {
            break;
        }
        case PriorMeanFunction::LINEAR: {
            parameters[ip] = priorConstant_;
            ++ip;

            for (uint_t j = 0; j < d; ++j) {
                parameters[ip] = priorLinearCoefficients_[j];
                ++ip;
            }
            break;
        }
    };

    assert(ip == parameters.size());

    return parameters;
}

void GPz::loadParametersArray_(const Vec1d& parameters) {
    const uint_t m = numberPseudoInputs_;
    const uint_t d = numberFeatures_;

    // Pseudo input position
    uint_t ip = 0;
    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j < d; ++j) {
        pseudoInputPositions_(i,j) = parameters[ip];
        ++ip;
    }

    // Pseudo input relevances
    for (uint_t i = 0; i < m; ++i) {
        pseudoInputRelevances_[i] = parameters[ip];
        ++ip
    }

    // Pseudo input covariances
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            double covariance = parameters_[ip];
            ++ip;

            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = 0; k < d; ++k) {
                pseudoInputCovariances_[i](j,k) = (k == j ? covariance : 0.0);
            }

            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            for (uint_t i = 0; i < m; ++i) {
                double covariance = parameters_[ip];
                ++ip;

                for (uint_t j = 0; j < d; ++j)
                for (uint_t k = 0; k < d; ++k) {
                    pseudoInputCovariances_[i](j,k) = (k == j ? covariance : 0.0);
                }
            }
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            for (uint_t j = 0; j < d; ++j) {
                double covariance = parameters_[ip];
                ++ip;

                for (uint_t i = 0; i < m; ++i)
                for (uint_t k = 0; k < d; ++k) {
                    pseudoInputCovariances_[i](j,k) = (k == j ? covariance : 0.0);
                }
            }
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j) {
                double covariance = parameters[ip];
                ++ip;

                for (uint_t k = 0; k < d; ++k) {
                    pseudoInputCovariances_[i](j,k) = (k == j ? covariance : 0.0);
                }
            }
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                double covariance = parameters[ip];
                ++ip;

                for (uint_t i = 0; i < d; ++i) {
                    pseudoInputCovariances_[i](j,k) = covariance;
                    pseudoInputCovariances_[i](k,j) = covariance;
                }
            }
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                double covariance = parameters[ip];
                ++ip;

                pseudoInputCovariances_[i](j,k) = covariance;
                pseudoInputCovariances_[i](k,j) = covariance;
            }
            break;
        }
    }

    // Output error parametrization
    switch (outputErrorType_) {
        case OutputErrorType::CONSTANT: {
            errorConstant_ = parameters[ip];
            ++ip;
            break;
        }
        case OutputErrorType::HETEROSCEDASTIC: {
            // Constant
            errorConstant_ = parameters[ip];
            ++ip;

            // Weights
            for (uint_t i = 0; i < m; ++i) {
                errorWeights_[i]; = parameters[ip];
                ++ip;
            }

            // Relevance
            for (uint_t i = 0; i < m; ++i) {
                errorRelevances_[i] = parameters[ip];
                ++ip;
            }

            break;
        }
    }

    // Prior mean function
    switch (priorMean_) {
        case PriorMeanFunction::ZERO: {
            break;
        }
        case PriorMeanFunction::LINEAR: {
            priorConstant_ = parameters[ip];
            ++ip;

            for (uint_t j = 0; j < d; ++j) {
                priorLinearCoefficients_[j] = parameters[ip];
                ++ip;
            }
            break;
        }
    };

    assert(ip == parameters.size());
}

void GPz::resizeArrays_() const {
    const uint_t m = numberPseudoInputs_;
    const uint_t d = numberFeatures_;

    pseudoInputPositions_.resize(m,d);
    pseudoInputRelevances_.resize(m);
    pseudoInputCovariances_.resize(m);
    for (uint_t i = 0; i < m; ++i) {
        pseudoInputCovariances_[i].resize(d,d);
    }

    errorConstant_ = 0.0;
    errorWeights_.resize(m);
    errorRelevances_.resize(m);

    priorConstant_ = 0.0;
    priorLinearCoefficients_.resize(d);
}

void GPz::reset_() {
    pseudoInputPositions_.clear();
    pseudoInputRelevances_.clear();
    pseudoInputCovariances_.clear();

    errorConstant_ = 0.0;
    errorWeights_.clear();
    errorRelevances_.clear();

    priorConstant_ = 0.0;
    priorLinearCoefficients_.clear();
}

void GPz::initializePseudoInputs_() {
    // TODO: copy the GPz MatLab implementation for this

    MapMat2d pseudoPositions = getPseudoInputPositions_();
    for (uint_t i = 0; i < m; ++i)
    for (uint_t k = 0; k < d; ++k) {
        // TODO: set this to better initial value
        pseudoPositions(i,k) = 0.0;
    }
}

void GPz::initializePseudoInputRelevances_() {
    // TODO: copy the GPz MatLab implementation for this

    // Pseudo input relevance
    for (uint_t i = 0, ip = indexPseudoRelevance_; i < m; ++i, ++ip) {
        parameters_[ip] = 1.0;
    }
}

Mat2d GPz::initializeCovariancesFillLinear_(const Vec2d& x) {
    // TODO: copy the GPz MatLab implementation for this

}

Vec1d GPz::initializeCovariancesMakeGamma_(const Vec2d& x) {
    const uint_t m = numberPseudoInputs_;
    const uint_t d = numberFeatures_;
    const uint_t n = x.rows();

    Mat2d linearInputs = initializeCovariancesFillLinear_(x);

    Vec1d gamma(m);
    double factor = 0.5*pow(m, 1.0/d);
    for (uint_t i = 0; i < m; ++i) {
        double meanSquaredDist = 0.0;
        for (uint_t j = 0; j < n; ++j)
        for (uint_t k = 0; k < d; ++k) {
            double d = pseudoInputPositions_(i,k) - linearInputs(j,k);
            me += d*d;
        }

        meanSquaredDist /= n;

        gamma[i] = sqrt(factor/meanSquaredDist);
    }

    return gamma;
}

void GPz::initializeCovariances_(const Vec2d& x) {
    const uint_t m = numberPseudoInputs_;
    const uint_t d = numberFeatures_;

    // Pseudo input covariance
    Vec1d gamma = initializeCovariancesMakeGamma_(x);
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            parameters_[indexPseudoCovariance_] = gamma.mean();
            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            for (uint_t i = 0, ip = indexPseudoCovariance_; i < m; ++i, ++ip) {
                parameters_[ip] = gamma[i];
            }
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            double mean_gamma = gamma.mean();
            for (uint_t i = 0, ip = indexPseudoCovariance_; i < d; ++i, ++ip) {
                parameters_[ip] = mean_gamma;
            }
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            uint_t ip = indexPseudoCovariance_;
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; i < d; ++j) {
                parameters_[ip] = gamma[i];
                ++ip;
            }
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            double mean_gamma = gamma.mean();
            uint_t ip = indexPseudoCovariance_;
            // Diagonal
            for (uint_t i = 0; i < d; ++i, ++ip) {
                parameters_[ip] = mean_gamma;
            }
            // Off-diagonal
            for (uint_t i = 0; i < d*(d-1)/2; ++i, ++ip) {
                parameters_[ip] = 0.0;
            }
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            uint_t ip = indexPseudoCovariance_;
            for (uint_t i = 0; i < m; ++i) {
                // Diagonal
                for (uint_t j = 0; j < d; ++j) {
                    parameters_[ip] = gamma[i];
                    ++ip;
                }
                // Off-diagonal
                for (uint_t j = 0; j < d*(d-1)/2; ++j) {
                    parameters_[ip] = 0.0;
                    ++ip;// TODO: copy the GPz MatLab implementation for this

                }
            }
            break;
        }
    }
}

void GPz::initializeErrors_() {
    // TODO: copy the GPz MatLab implementation for this

    const uint_t m = numberPseudoInputs_;

    switch (outputErrorType_) {
        case OutputErrorType::CONSTANT: {
            parameters_[indexError_] = 1.0;
            break;
        }
        case OutputErrorType::HETEROSCEDASTIC: {
            parameters_[indexError_] = 1.0;
            for (uint_t i = 0, ip = indexErrorWeight_; i < m; ++i, ++ip) {
                parameters_[ip] = 1.0;
            }
            for (uint_t i = 0, ip = indexErrorRelevance_; i < m; ++i, ++ip) {
                parameters_[ip] = 1.0;
            }
            break;
        }
    }
}

void GPz::initializePriors_() {
    // TODO: copy the GPz MatLab implementation for this

    const uint_t d = numberFeatures_;

    switch (priorMean_) {
        case PriorMeanFunction::ZERO: {
            break;
        }
        case PriorMeanFunction::LINEAR: {
            uint_t ip = indexPriorMean_;
            // Constant
            parameters_[ip] = 0.0;
            ++ip;
            // Weights
            for (uint_t i = 0; i < d; ++i, ++ip) {
                parameters_[ip] = 1.0;
            }
            break;
        }
    };
}

void GPz::initializeFit_(const Vec2d& x, const Vec2d& xe, const Vec1d& y) {
    // Create arrays, matrices, etc.
    setNumberOfFeatures(x.cols());
    updateNumberParameters_();
    resizeArrays_();

    // Set initial values for hyper-parameters
    initializePseudoInputs_();
    initializePseudoInputRelevances_();
    initializeCovariances_(x);
    initializeErrors_();
    initializePriors_();
}

// =======================
// Internal functions: fit
// =======================

void GPz::updateLikelihood_(Minimize::FunctionOutput requested) {

}

// =============================
// Configuration getters/setters
// =============================

void GPz::setNumberOfPseudoInputs(uint_t num) {
    if (num != numberPseudoInputs_) {
        numberPseudoInputs_ = num;
        reset_();
    }
}

uint_t GPz::getNumberOfPseudoInputs() const {
    return numberPseudoInputs_;
}

void GPz::setPriorMeanFunction(PriorMeanFunction newFunction) {
    if (newFunction != priorMean_) {
        priorMean_ = newFunction;
        reset_();
    }
}

GPz::PriorMeanFunction GPz::getPriorMeanFunction() const {
    return priorMean_;
}

void GPz::setNumberOfFeatures(uint_t num) {
    if (num != numberFeatures_) {
        numberFeatures_ = num;
        reset_();
    }
}

uint_t GPz::getNumberOfFeatures() const {
    return numberFeatures_;
}

void GPz::setWeightingScheme(WeightingScheme scheme) {
    weightingScheme_ = scheme;
}

GPz::WeightingScheme GPz::getWeightingScheme() const {
    return weightingScheme_;
}

// =====================
// Fit/training function
// =====================

void GPz::fit(const Vec2d& x, const Vec2d& xe, const Vec1d& y) {
    assert(xe.empty() || (xe.rows() == x.rows() && xe.cols() == x.cols()));

    initializeFit_(x, xe, y);

    Minimize::minimizeBFGS(options, makeParameterArray_(),
        [this](const Vec1d& parameters, Minimize::FunctionOutput requested) {

            if (requested != Minimize::FunctionOutput::METRIC_VALID) {
                loadParametersArray_(parameters);
            }

            updateLikelihood_(requested);

            Vec1d result;

            if (requested == Minimize::FunctionOutput::METRIC_VALID) {
                result.resize(1);
                result[0] = logLikelihoodValid_;
            } else {
                result.resize(1+numberParameters_);

                if (requested == Minimize::FunctionOutput::ALL ||
                    requested == Minimize::FunctionOutput::METRIC_TRAIN) {
                    result[0] = logLikelihood_;
                }

                if (requested == Minimize::FunctionOutput::ALL ||
                    requested == Minimize::FunctionOutput::DERIVATIVES) {
                    for (uint_t i = 0; i < numberParameters_; ++i) {
                        result[1+i] = derivatives_[i];
                    }
                }
            }

            return result;
        }
    );
}

// =================================
// Fit/training result getter/setter
// =================================

Vec1d GPz::getParameters() const {
    return makeParameterArray_();
}

void GPz::setParameters(const Vec1d& newParameters) {
    loadParametersArray_(newParameters);
}

// ===================
// Prediction function
// ===================

Vec1d GPz::predict(const Vec2d& x, const Vec2d& xe) const {
    assert(x.cols() == numberFeatures_);
    assert(xe.empty() || (xe.rows() == x.rows() && xe.cols() == x.cols()));
    assert(!parameters_.empty());

    return Vec1d();
}

Vec1d GPz::predict(const Vec2d& x) const {
    return predict(x, Vec2d());
}

}  // namespace PHZ_GPz


