/**
 * @file PHZ_GPz/GPz.h
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

#ifndef _PHZ_GPZ_GPZ_H
#define _PHZ_GPZ_GPZ_H

#include "PHZ_GPz/EigenWrapper.h"
#include <vector>

namespace PHZ_GPz {

/**
 * @class GPz
 * @brief Training and prediction using Gaussian Processes
 *
 */
class GPz {

public:

    // Enumerations for configuration

    /**
     * @brief Choice of prior mean function
     */
    enum class PriorMeanFunction {
        ZERO, LINEAR
    };

    /**
     * @brief Choice of covariance parametriation
     */
    enum class CovarianceType {
        GLOBAL_LENGTH,      // GPGL
        VARIABLE_LENGTH,    // GPVL
        GLOBAL_DIAGONAL,    // GPGD
        VARIABLE_DIAGONAL,  // GPVD
        GLOBAL_COVARIANCE,  // GPGC
        VARIABLE_COVARIANCE // GPVC
    };

    /**
     * @brief Choice of output noise parametrization
     */
    enum class OutputErrorType {
        CONSTANT,
        HETEROSCEDASTIC
    };

    /**
     * @brief Choice of training data weighting scheme
     */
    enum class WeightingScheme {
        NATURAL,
        ONE_OVER_ONE_PLUS_OUTPUT,
        BALANCED
    };

private:

    // =======================
    // Configuration variables
    // =======================

    uint_t            numberPseudoInputs_ = 100;
    PriorMeanFunction priorMean_ = PriorMeanFunction::LINEAR;
    CovarianceType    covarianceType_ = CovarianceType::VARIABLE_COVARIANCE;
    OutputErrorType   outputErrorType_ = OutputErrorType::HETEROSCEDASTIC;
    WeightingScheme   weightingScheme_ = WeightingScheme::BALANCED;
    double            balancedWeightingBinSize_ = 0.1;

    // ==================
    // Indexing variables
    // ==================

    uint_t numberFeatures_ = 0;
    uint_t numberParameters_ = 0;

    uint_t indexPseudoPosition_ = 0;
    uint_t indexPseudoRelevance_ = 0;
    uint_t indexPseudoCovariance_ = 0;
    uint_t indexError_ = 0;
    uint_t indexErrorWeight_ = 0;
    uint_t indexErrorRelevance_ = 0;
    uint_t indexPriorMean_ = 0;

    // ================
    // Hyper-parameters
    // ================

    Mat2d              pseudoInputPositions_;
    Mat1d              pseudoInputRelevances_;
    std::vector<Mat2d> pseudoInputCovariances_;
    double             errorConstant_ = 0.0;
    Mat1d              errorWeights_;
    Mat1d              errorRelevances_;
    double             priorConstant_ = 0.0;
    Mat1d              priorLinearCoefficients_;

    // ======================
    // Minimization variables
    // ======================

    double logLikelihood_ = 0.0;
    double logLikelihoodValid_ = 0.0;
    Mat1d  derivatives_;

private:

    // =========================
    // Internal functions: setup
    // =========================

    void updateNumberParameters_() {
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

    Vec1d makeParameterArray_() const {
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

    void loadParametersArray_(const Vec1d& parameters) {
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

    void resizeArrays_() const {
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

    void reset_() {
        pseudoInputPositions_.clear();
        pseudoInputRelevances_.clear();
        pseudoInputCovariances_.clear();

        errorConstant_ = 0.0;
        errorWeights_.clear();
        errorRelevances_.clear();

        priorConstant_ = 0.0;
        priorLinearCoefficients_.clear();
    }

    void initializePseudoInputs_() {
        // TODO: copy the GPz MatLab implementation for this

        MapMat2d pseudoPositions = getPseudoInputPositions_();
        for (uint_t i = 0; i < m; ++i)
        for (uint_t k = 0; k < d; ++k) {
            // TODO: set this to better initial value
            pseudoPositions(i,k) = 0.0;
        }
    }

    void initializePseudoInputRelevances_() {
        // TODO: copy the GPz MatLab implementation for this

        // Pseudo input relevance
        for (uint_t i = 0, ip = indexPseudoRelevance_; i < m; ++i, ++ip) {
            parameters_[ip] = 1.0;
        }
    }

    Mat2d initializeCovariancesFillLinear_(const Vec2d& x) {
        // TODO: copy the GPz MatLab implementation for this

    }

    Vec1d initializeCovariancesMakeGamma_(const Vec2d& x) {
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

    void initializeCovariances_(const Vec2d& x) {
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

    void initializeErrors_() {
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

    void initializePriors_() {
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

    void initializeFit_(const Vec2d& x, const Vec2d& xe, const Vec1d& y) {
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

    void updateLikelihood_(Minimize::FunctionOutput requested) {

    }

public:

    // ==========================
    // Constructors and operators
    // ==========================

    /**
     * @brief Default constructor
     */
    GPz() = default;

    /**
     * @brief Destructor
     */
    ~GPz() = default;

    /**
     * @brief Copy constructor
     */
    GPz(const GPz&) = default;

    /**
     * @brief Move constructor
     */
    GPz(GPz&&) = default;

    /**
     * @brief Copy assignment operator
     */
    GPz& operator=(const GPz&) = default;

    /**
     * @brief Move assignment operator
     */
    GPz& operator=(GPz&&) = default;

    // =============================
    // Configuration getters/setters
    // =============================

    void setNumberOfPseudoInputs(uint_t num);

    uint_t getNumberOfPseudoInputs() const;

    void setPriorMeanFunction(PriorMeanFunction newFunction);

    PriorMeanFunction getPriorMeanFunction() const;

    void setNumberOfFeatures(uint_t num);

    uint_t getNumberOfFeatures() const;

    void setWeightingScheme(WeightingScheme scheme);

    WeightingScheme getWeightingScheme() const;

    // =====================
    // Fit/training function
    // =====================

    void fit(const Vec2d& x, const Vec2d& xe, const Vec1d& y);

    // =================================
    // Fit/training result getter/setter
    // =================================

    Vec1d getParameters() const;

    void setParameters(const Vec1d& newParameters);

    // ===================
    // Prediction function
    // ===================

    Vec1d predict(const Vec2d& x, const Vec2d& xe) const;

    Vec1d predict(const Vec2d& x) const;

};  // End of GPz class

}  // namespace PHZ_GPz


#endif
