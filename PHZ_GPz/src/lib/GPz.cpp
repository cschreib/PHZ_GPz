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

// ====================================
// Internal functions: hyper-parameters
// ====================================

Vec1d GPz::makeParameterArray_(const HyperParameters& inputParams) const  {
    Vec1d outputParams(numberParameters_);

    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    // Pseudo input position
    uint_t ip = 0;
    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j < d; ++j) {
        outputParams[ip] = inputParams.basisFunctionPositions(i,j);
        ++ip;
    }

    // Pseudo input relevances
    for (uint_t i = 0; i < m; ++i) {
        outputParams[ip] = inputParams.basisFunctionRelevances[i];
        ++ip
    }

    // Pseudo input covariances
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            outputParams[ip] = inputParams.basisFunctionCovariances[0](0,0);
            ++ip;
            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            for (uint_t i = 0; i < m; ++i) {
                outputParams[ip] = inputParams.basisFunctionCovariances[i](0,0);
                ++ip;
            }
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            for (uint_t j = 0; j < d; ++j) {
                outputParams[ip] = inputParams.basisFunctionCovariances[0](j,j);
                ++ip;
            }
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j) {
                outputParams[ip] = inputParams.basisFunctionCovariances[i](j,j);
                ++ip;
            }
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                outputParams[ip] = inputParams.basisFunctionCovariances[0](j,k);
                ++ip;
            }
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                outputParams[ip] = inputParams.basisFunctionCovariances[i](j,k);
                ++ip;
            }
            break;
        }
    }

    // Output error parametrization
    switch (outputUncertaintyType_) {
        case OutputUncertaintyType::UNIFORM: {
            outputParams[ip] = inputParams.uncertaintyConstant;
            ++ip;
            break;
        }
        case OutputUncertaintyType::INPUT_DEPENDENT: {
            // Constant
            outputParams[ip] = inputParams.uncertaintyConstant;
            ++ip;

            // Weights
            for (uint_t i = 0; i < m; ++i) {
                outputParams[ip] = inputParams.uncertaintyBasisWeights[i];
                ++ip;
            }

            // Relevance
            for (uint_t i = 0; i < m; ++i) {
                outputParams[ip] = inputParams.uncertaintyBasisRelevances[i];
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
            outputParams[ip] = inputParams.priorConstant;
            ++ip;

            for (uint_t j = 0; j < d; ++j) {
                outputParams[ip] = inputParams.priorLinearCoefficients[j];
                ++ip;
            }
            break;
        }
    };

    assert(ip == outputParams.size());

    return outputParams;
}

void GPz::loadParametersArray_(const Vec1d& inputParams, HyperParameters& outputParams) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    // Pseudo input position
    uint_t ip = 0;
    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j < d; ++j) {
        outputParams.basisFunctionPositions(i,j) = inputParams[ip];
        ++ip;
    }

    // Pseudo input relevances
    for (uint_t i = 0; i < m; ++i) {
        outputParams.basisFunctionRelevances[i] = inputParams[ip];
        ++ip
    }

    // Pseudo input covariances
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            double covariance = inputParams[ip];
            ++ip;

            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = 0; k < d; ++k) {
                outputParams.basisFunctionCovariances[i](j,k) = (k == j ? covariance : 0.0);
            }

            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            for (uint_t i = 0; i < m; ++i) {
                double covariance = inputParams[ip];
                ++ip;

                for (uint_t j = 0; j < d; ++j)
                for (uint_t k = 0; k < d; ++k) {
                    outputParams.basisFunctionCovariances[i](j,k) = (k == j ? covariance : 0.0);
                }
            }
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            for (uint_t j = 0; j < d; ++j) {
                double covariance = inputParams[ip];
                ++ip;

                for (uint_t i = 0; i < m; ++i)
                for (uint_t k = 0; k < d; ++k) {
                    outputParams.basisFunctionCovariances[i](j,k) = (k == j ? covariance : 0.0);
                }
            }
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j) {
                double covariance = inputParams[ip];
                ++ip;

                for (uint_t k = 0; k < d; ++k) {
                    outputParams.basisFunctionCovariances[i](j,k) = (k == j ? covariance : 0.0);
                }
            }
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                double covariance = inputParams[ip];
                ++ip;

                for (uint_t i = 0; i < d; ++i) {
                    outputParams.basisFunctionCovariances[i](j,k) = covariance;
                    outputParams.basisFunctionCovariances[i](k,j) = covariance;
                }
            }
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = j; k < d; ++k) {
                double covariance = inputParams[ip];
                ++ip;

                outputParams.basisFunctionCovariances[i](j,k) = covariance;
                outputParams.basisFunctionCovariances[i](k,j) = covariance;
            }
            break;
        }
    }

    // Output error parametrization
    switch (outputUncertaintyType_) {
        case OutputUncertaintyType::UNIFORM: {
            outputParams.uncertaintyConstant = inputParams[ip];
            ++ip;
            break;
        }
        case OutputUncertaintyType::INPUT_DEPENDENT: {
            // Constant
            outputParams.uncertaintyConstant = inputParams[ip];
            ++ip;

            // Weights
            for (uint_t i = 0; i < m; ++i) {
                outputParams.uncertaintyBasisWeights[i]; = inputParams[ip];
                ++ip;
            }

            // Relevance
            for (uint_t i = 0; i < m; ++i) {
                outputParams.uncertaintyBasisRelevances[i] = inputParams[ip];
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
            priorConstant_ = inputParams[ip];
            ++ip;

            for (uint_t j = 0; j < d; ++j) {
                priorLinearCoefficients_[j] = inputParams[ip];
                ++ip;
            }
            break;
        }
    };

    assert(ip == inputParams.size());
}

void GPz::resizeHyperParameters_(HyperParameters& params) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    params.basisFunctionPositions.resize(m,d);
    params.basisFunctionRelevances.resize(m);
    params.basisFunctionCovariances.resize(m);
    for (uint_t i = 0; i < m; ++i) {
        params.basisFunctionCovariances[i].resize(d,d);
    }

    // TODO: Save memory: basisFunctionCovariances only needs one element
    // when the covariance type is any of the GLOBAL_* types.

    params.uncertaintyBasisWeights.resize(m);
    params.uncertaintyBasisRelevances.resize(m);

    params.priorLinearCoefficients.resize(d);
}


// ==================================
// Internal functions: initialization
// ==================================

void GPz::updateNumberParameters_() {
    // Pseudo input position
    indexBasisPosition_ = 0;
    numberParameters_ += numberBasisFunctions_*numberFeatures_;

    // Pseudo input relevance
    indexBasisRelevance_ = numberParameters_;
    numberParameters_ += numberBasisFunctions_;

    // Pseudo input covariance
    indexBasisCovariance_ = numberParameters_;
    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH: {
            numberParameters_ += 1;
            break;
        }
        case CovarianceType::VARIABLE_LENGTH: {
            numberParameters_ += numberBasisFunctions_;
            break;
        }
        case CovarianceType::GLOBAL_DIAGONAL: {
            numberParameters_ += numberFeatures_;
            break;
        }
        case CovarianceType::VARIABLE_DIAGONAL : {
            numberParameters_ += numberFeatures_*numberBasisFunctions_;
            break;
        }
        case CovarianceType::GLOBAL_COVARIANCE : {
            numberParameters_ += numberFeatures_*(numberFeatures_ + 1)/2;
            break;
        }
        case CovarianceType::VARIABLE_COVARIANCE : {
            numberParameters_ += numberFeatures_*(numberFeatures_ + 1)/2*numberBasisFunctions_;
            break;
        }
    }

    // Output error parametrization
    indexError_ = numberParameters_;
    switch (outputUncertaintyType_) {
        case OutputUncertaintyType::UNIFORM: {
            numberParameters_ += 1;
            break;
        }
        case OutputUncertaintyType::INPUT_DEPENDENT: {
            // Constant
            numberParameters_ += 1;

            // Weights
            indexErrorWeight_ = numberParameters_;
            numberParameters_ += numberBasisFunctions_;

            // Relevance
            indexErrorRelevance_ = numberParameters_;
            numberParameters_ += numberBasisFunctions_;
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

void GPz::resizeArrays_() {
    resizeHyperParameters_(parameters_);
    resizeHyperParameters_(derivatives_);
}

void GPz::reset_() {
    parameters_ = HyperParameters{};
    derivatives_ = HyperParameters{};
}

void GPz::initializeBasisFunctions_() {
    // TODO: copy the GPz MatLab implementation for this

    for (uint_t i = 0; i < m; ++i)
    for (uint_t k = 0; k < d; ++k) {
        // TODO: set this to better initial value
        parameters_.basisFunctionPositions(i,k) = 0.0;
    }
}

void GPz::initializeBasisFunctionRelevances_() {
    // TODO: copy the GPz MatLab implementation for this

    for (uint_t i = 0; i < m; ++i) {
        parameters_.basisFunctionRelevances[i] = 1.0;
    }
}

Mat2d GPz::initializeCovariancesFillLinear_(const Vec2d& x) {
    // TODO: placeholder

    return Mat2d{};
}

Vec1d GPz::initializeCovariancesMakeGamma_(const Vec2d& x) {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;
    const uint_t n = x.rows();

    Mat2d linearInputs = initializeCovariancesFillLinear_(x);

    Vec1d gamma(m);
    double factor = 0.5*pow(m, 1.0/d);
    for (uint_t i = 0; i < m; ++i) {
        double meanSquaredDist = 0.0;
        for (uint_t j = 0; j < n; ++j)
        for (uint_t k = 0; k < d; ++k) {
            double d = parameters_.basisFunctionPositions(i,k) - linearInputs(j,k);
            me += d*d;
        }

        meanSquaredDist /= n;

        gamma[i] = sqrt(factor/meanSquaredDist);
    }

    return gamma;
}

void GPz::initializeCovariances_(const Vec2d& x) {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    // Compute some statistics from training set
    Vec1d gamma = initializeCovariancesMakeGamma_(x);

    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH:
        case CovarianceType::GLOBAL_DIAGONAL:
        case CovarianceType::GLOBAL_COVARIANCE: {
            double mean_gamma = gamma.mean();

            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = 0; k < d; ++k) {
                parameters_.basisFunctionCovariances_[i](j,k) = (j == k ? mean_gamma : 0.0);
            }

            break;
        }
        case CovarianceType::VARIABLE_LENGTH:
        case CovarianceType::VARIABLE_DIAGONAL:
        case CovarianceType::VARIABLE_COVARIANCE: {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = 0; k < d; ++k) {
                parameters_.basisFunctionCovariances_[i](j,k) = (j == k ? gamma[i] : 0.0);
            }

            break;
        }
    }
}

void GPz::initializeErrors_() {
    // TODO: copy the GPz MatLab implementation for this

    const uint_t m = numberBasisFunctions_;

    switch (outputUncertaintyType_) {
        case OutputUncertaintyType::UNIFORM: {
            parameters_.uncertaintyConstant = 1.0;
            break;
        }
        case OutputUncertaintyType::INPUT_DEPENDENT: {
            parameters_.uncertaintyConstant = 1.0;
            for (uint_t i = 0; i < m; ++i) {
                parameters_.uncertaintyBasisWeights[i] = 1.0;
                parameters_.uncertaintyBasisRelevances[i] = 1.0;
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
            // Constant
            parameters_.priorConstant = 0.0;
            // Weights
            for (uint_t i = 0; i < d; ++i) {
                parameters_.priorLinearCoefficients[i] = 1.0;
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
    initializeBasisFunctions_();
    initializeBasisFunctionRelevances_();
    initializeCovariances_(x);
    initializeErrors_();
    initializePriors_();
}

// =======================
// Internal functions: fit
// =======================

void GPz::updateLikelihood_(Minimize::FunctionOutput requested) {
    // TODO: placeholder
}


// ==============================
// Internal functions: prediction
// ==============================

Vec1d GPz::predict_(const Vec2d& x, const Vec2d& xe) {
    // TODO: placeholder

    return Vec1d{};
}


// =============================
// Configuration getters/setters
// =============================

void GPz::setNumberOfBasisFunctions(uint_t num) {
    if (num != numberBasisFunctions_) {
        numberBasisFunctions_ = num;
        reset_();
    }
}

uint_t GPz::getNumberOfBasisFunctions() const {
    return numberBasisFunctions_;
}

void GPz::setPriorMeanFunction(PriorMeanFunction newFunction) {
    if (newFunction != priorMean_) {
        priorMean_ = newFunction;
        reset_();
    }
}

PriorMeanFunction GPz::getPriorMeanFunction() const {
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

WeightingScheme GPz::getWeightingScheme() const {
    return weightingScheme_;
}

// =====================
// Fit/training function
// =====================

void GPz::fit(const Vec2d& x, const Vec2d& xe, const Vec1d& y) {
    // Check inputs are consistent
    assert(xe.empty() || (xe.rows() == x.rows() && xe.cols() == x.cols()));

    // Setup the fit, initialize arrays, etc.
    initializeFit_(x, xe, y);

    // Build vector with initial values for hyper-parameter
    Vec1d initialValues = makeParameterArray_(parameters_);

    // Use BFGS for minimization
    Minimize::minimizeBFGS(options, initialValues,
        [this](const Vec1d& vectorParameters, Minimize::FunctionOutput requested) {

            if (requested != Minimize::FunctionOutput::METRIC_VALID) {
                // Load new parameters
                loadParametersArray_(vectorParameters, parameters_);
            }

            // Compute/update the requested quantities
            updateLikelihood_(requested);

            Vec1d result;

            if (requested == Minimize::FunctionOutput::METRIC_VALID) {
                // Return only the log likelihood of the validation set
                result.resize(1);
                result[0] = logLikelihoodValid_;
            } else {
                result.resize(1+numberParameters_);

                if (requested == Minimize::FunctionOutput::ALL ||
                    requested == Minimize::FunctionOutput::METRIC_TRAIN) {
                    // Return log likelihood of the training set
                    result[0] = logLikelihood_;
                }

                if (requested == Minimize::FunctionOutput::ALL ||
                    requested == Minimize::FunctionOutput::DERIVATIVES) {
                    // Return derivatives of log likelihood with respect to hyper-parameters
                    Vec1d vectorDerivatives = makeParameterArray_(derivatives_);
                    for (uint_t i = 0; i < numberParameters_; ++i) {
                        result[1+i] = vectorDerivatives[i];
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
    return makeParameterArray_(parameters_);
}

void GPz::setParameters(const Vec1d& newParameters) {
    loadParametersArray_(newParameters, parameters_);
}

// ===================
// Prediction function
// ===================

Vec1d GPz::predict(const Vec2d& x, const Vec2d& xe) const {
    // Check input is consistent
    assert(x.cols() == numberFeatures_);
    assert(xe.empty() || (xe.rows() == x.rows() && xe.cols() == x.cols()));

    // Check that we have a usable set of parameters to make predictions
    assert(!parameters_.empty());

    return predict_(x, xe);
}

Vec1d GPz::predict(const Vec2d& x) const {
    return predict(x, Vec2d());
}

}  // namespace PHZ_GPz


