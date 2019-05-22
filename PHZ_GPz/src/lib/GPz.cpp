/**
 * @file src/lib/GPz.cpp
 * @date 11/29/18
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

#include "PHZ_GPz/GPz.h"
#include "PHZ_GPz/LBFGS.h"
#ifndef NO_GSL
#include "PHZ_GPz/GSLWrapper.h"
#endif

#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <gperftools/profiler.h>

namespace PHZ_GPz {

// ====================================
// Internal functions: hyper-parameters
// ====================================

Vec1d GPz::makeParameterArray_(const GPzHyperParameters& inputParams) const  {
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
        outputParams[ip] = inputParams.basisFunctionLogRelevances[i];
        ++ip;
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
            outputParams[ip] = inputParams.logUncertaintyConstant;
            ++ip;
            break;
        }
        case OutputUncertaintyType::INPUT_DEPENDENT: {
            // Constant
            outputParams[ip] = inputParams.logUncertaintyConstant;
            ++ip;

            // Weights
            for (uint_t i = 0; i < m; ++i) {
                outputParams[ip] = inputParams.uncertaintyBasisWeights[i];
                ++ip;
            }

            // Relevance
            for (uint_t i = 0; i < m; ++i) {
                outputParams[ip] = inputParams.uncertaintyBasisLogRelevances[i];
                ++ip;
            }

            break;
        }
    }

    assert(ip == static_cast<uint_t>(outputParams.size()) && "bug in parameter array (make)");

    return outputParams;
}

void GPz::loadParametersArray_(const Vec1d& inputParams, GPzHyperParameters& outputParams) const {
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
        outputParams.basisFunctionLogRelevances[i] = inputParams[ip];
        ++ip;
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
            outputParams.logUncertaintyConstant = inputParams[ip];
            ++ip;
            break;
        }
        case OutputUncertaintyType::INPUT_DEPENDENT: {
            // Constant
            outputParams.logUncertaintyConstant = inputParams[ip];
            ++ip;

            // Weights
            for (uint_t i = 0; i < m; ++i) {
                outputParams.uncertaintyBasisWeights[i] = inputParams[ip];
                ++ip;
            }

            // Relevance
            for (uint_t i = 0; i < m; ++i) {
                outputParams.uncertaintyBasisLogRelevances[i] = inputParams[ip];
                ++ip;
            }

            break;
        }
    }

    assert(ip == static_cast<uint_t>(inputParams.size()) && "bug in parameter array (load)");
}

void GPz::resizeHyperParameters_(GPzHyperParameters& params) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    params.basisFunctionPositions.setZero(m,d);
    params.basisFunctionLogRelevances.setZero(m);
    params.basisFunctionCovariances.resize(m);
    for (uint_t i = 0; i < m; ++i) {
        params.basisFunctionCovariances[i].setZero(d,d);
    }

    // TODO: Save memory: basisFunctionCovariances only needs one element
    // when the covariance type is any of the GLOBAL_* types.

    params.uncertaintyBasisWeights.setZero(m);
    params.uncertaintyBasisLogRelevances.setZero(m);
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

    if (verbose_) {
        std::cout << "the fit has " << numberParameters_ << " parameters" << std::endl;
        std::cout << " - " << indexBasisRelevance_ - indexBasisPosition_ << " BF positions" << std::endl;
        std::cout << " - " << indexBasisCovariance_ - indexBasisRelevance_<< " BF relevances" << std::endl;
        std::cout << " - " << indexError_ - indexBasisCovariance_ << " BF covariances" << std::endl;
        std::cout << " - " << numberParameters_ - indexError_ << " for output error" << std::endl;
    }
}

void GPz::resizeArrays_() {
    const uint_t d = numberFeatures_;

    resizeHyperParameters_(parameters_);
    resizeHyperParameters_(derivatives_);

    if (normalizationScheme_ == NormalizationScheme::WHITEN) {
        featureMean_.setZero(d);
        featureSigma_.setZero(d);
    }

    featurePCAMean_.setZero(d);
    featurePCASigma_.setZero(d,d);
    featurePCABasisVectors_.setZero(d,d);

    if (verbose_) {
        std::cout << "allocated memory for fit parameters" << std::endl;
    }
}

void GPz::reset_() {
    parameters_ = GPzHyperParameters{};
    derivatives_ = GPzHyperParameters{};

    featureMean_.resize(0);
    featureSigma_.resize(0);
    outputMean_ = 0.0;

    missingCache_.clear();

    featurePCAMean_.resize(0);
    featurePCASigma_.resize(0,0);
    featurePCABasisVectors_.resize(0,0);

    decorrelationCoefficients_.resize(0);

    inputTrain_.resize(0,0);
    inputErrorTrain_.resize(0,0);
    outputTrain_.resize(0);
    weightTrain_.resize(0);
    missingTrain_.resize(0);
    inputValid_.resize(0,0);
    inputErrorValid_.resize(0,0);
    outputValid_.resize(0);
    weightValid_.resize(0);
    missingValid_.resize(0);

    trainBasisFunctions_.resize(0,0);
    trainOutputLogError_.resize(0);
    validBasisFunctions_.resize(0,0);
    validOutputLogError_.resize(0);
    modelWeights_.resize(0);
    modelInvCovariance_.resize(0,0);
    modelInputPrior_.resize(0);
}

void GPz::applyInputNormalization_(Mat2d& input, Mat2d& inputError) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // Transform error to variance
    if (inputError.rows() != 0) {
        for (uint_t i = 0; i < n; ++i)
        for (uint_t j = 0; j < d; ++j) {
            inputError(i,j) = inputError(i,j)*inputError(i,j);
        }
    }

    // Apply normalization scheme
    if (normalizationScheme_ == NormalizationScheme::WHITEN) {
        for (uint_t i = 0; i < n; ++i)
        for (uint_t j = 0; j < d; ++j) {
            input(i,j) = (input(i,j) - featureMean_[j])/featureSigma_[j];

            if (inputError.rows() != 0) {
                inputError(i,j) /= featureSigma_[j]*featureSigma_[j];
            }
        }
    }
}

void GPz::applyOutputNormalization_(const Mat2d& input, const Vec1i& /* missing */, Vec1d& output) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // TODO: not finalized implementation yet
    if (priorMean_ == PriorMeanFunction::LINEAR_PREPROCESS) {
        for (uint_t i = 0; i < n; ++i) {
            double pred = decorrelationCoefficients_[d];
            for (uint_t j = 0; j < d; ++j) {
                // TODO: fix this for missing variables
                pred += input(i,j)*decorrelationCoefficients_[j];
            }
            output[i] -= pred;
        }
    } else if (priorMean_ == PriorMeanFunction::CONSTANT_PREPROCESS) {
        output -= outputMean_;
    }
}

void GPz::restoreOutputNormalization_(const Mat2d& input, const Vec1i& /* missing */, GPzOutput& output) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // TODO: not finalized implementation yet
    if (priorMean_ == PriorMeanFunction::LINEAR_PREPROCESS) {
        for (uint_t i = 0; i < n; ++i) {
            double pred = decorrelationCoefficients_[d];
            for (uint_t j = 0; j < d; ++j) {
                // TODO: fix this for missing variables
                pred += input(i,j)*decorrelationCoefficients_[j];
            }
            output.value[i] += pred;
        }
    } else if (priorMean_ == PriorMeanFunction::CONSTANT_PREPROCESS) {
        output.value += outputMean_;
    }
}

void GPz::computeWhitening_(const Mat2d& input) {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    for (uint_t j = 0; j < d; ++j) {
        // For each feature, compute mean and standard deviation among all
        // data points and use the values to whiten the data (i.e., mean
        // of zero and standard deviation of unity).

        // Compute mean (filtering out missing data)
        featureMean_[j] = 0.0;
        uint_t count = 0;
        for (uint_t i = 0; i < n; ++i) {
            if (!std::isnan(input(i,j))) {
                featureMean_[j] += input(i,j);
                ++count;
            }
        }

        featureMean_[j] /= count;

        // Compute standard deviation (filtering out missing data)
        featureSigma_[j] = 0.0;
        count = 0;
        for (uint_t i = 0; i < n; ++i) {
            if (!std::isnan(input(i,j))) {
                double d = input(i,j) - featureMean_[j];
                featureSigma_[j] += d*d;
                ++count;
            }
        }

        featureSigma_[j] = sqrt(featureSigma_[j]/count);
    }
}

void GPz::computeLinearDecorrelation_(const Mat2d& input, const Mat2d& inputError,
    const Vec1d& output, const Vec1d& weight) {

    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // Fit output as linear combination of inputs plus constant

    // Compute model matrix and vector
    Mat2d modelInner(d+1, d+1);
    Mat1d modelObs(d+1);

    // X terms
    for (uint_t i = 0; i < d; ++i) {
        // Matrix
        for (uint_t j = i; j < d; ++j) {
            double sum = 0.0;
            for (uint_t k = 0; k < n; ++k) {
                // Skip missing data
                if (std::isnan(input(k,i) || std::isnan(input(k,j)))) continue;

                double temp = input(k,i)*input(k,j)*weight[k];
                if (inputError.rows() != 0) {
                    temp /= inputError(k,i)*inputError(k,j);
                }

                sum += temp;
            }

            modelInner(j,i) = modelInner(i,j) = sum;
        }

        // Vector
        double sum = 0.0;
        for (uint_t k = 0; k < n; ++k) {
            // Skip missing data
            if (std::isnan(input(k,i))) continue;

            double temp = input(k,i)*output[k]*weight[k];
            if (inputError.rows() != 0) {
                temp /= inputError(k,i)*inputError(k,i);
            }

            sum += temp;
        }

        modelObs[i] = sum;
    }

    // Constant term
    {
        uint_t i = d;

        // Matrix
        // Constant x X
        for (uint_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (uint_t k = 0; k < n; ++k) {
                // Skip missing data
                if (std::isnan(input(k,j))) continue;

                double temp = input(k,j)*weight[k];
                if (inputError.rows() != 0) {
                    temp /= inputError(k,j);
                }

                sum += temp;
            }

            modelInner(j,i) = modelInner(i,j) = sum;
        }

        // Constant x Constant
        modelInner(d,d) = weight.sum();

        // Vector
        modelObs[d] = (output*weight).sum();
    }

    // Compute Cholesky decomposition of model matrix
    Eigen::LDLT<Mat2d> cholesky(modelInner);

    // Compute best fit linear coefficients
    decorrelationCoefficients_ = cholesky.solve(modelObs);
}

void GPz::normalizeTrainingInputs_(Mat2d& input, Mat2d& inputError, const Vec1i& missing,
    Vec1d& output, const Vec1d& weight) {

    // Inputs

    if (normalizationScheme_ == NormalizationScheme::WHITEN) {
        computeWhitening_(input);
    }

    applyInputNormalization_(input, inputError);

    // Outputs

    if (priorMean_ == PriorMeanFunction::LINEAR_PREPROCESS) {
        computeLinearDecorrelation_(input, inputError, output, weight);
    } else if (priorMean_ == PriorMeanFunction::CONSTANT_PREPROCESS) {
        outputMean_ = output.mean();
    }

    applyOutputNormalization_(input, missing, output);
}

void GPz::eraseInvalidTrainData_(Mat2d& input, Mat2d& inputError, Vec1d& output) const {
    uint_t n = input.rows();
    const uint_t d = numberFeatures_;
    const bool noError = inputError.rows() == 0;

    std::vector<uint_t> valid;
    valid.reserve(n);

    for (uint_t i = 0; i < n; ++i) {
        bool good = false;

        if (std::isfinite(output[i])) {
            for (uint_t k = 0; k < d; ++k) {
                if (std::isfinite(input(i,k)) && (noError || std::isfinite(inputError(i,k)))) {
                    good = true;
                    break;
                }
            }
        }

        if (good) {
            valid.push_back(i);
        }
    }

    if (valid.size() != n) {
        if (verbose_) {
            std::cout << "removing " << n - valid.size() << " invalid inputs" << std::endl;
        }

        Mat2d oldInput = std::move(input);
        Mat2d oldError = std::move(inputError);
        Mat1d oldOutput = std::move(output);

        n = valid.size();

        output.resize(n);
        input.resize(n,d);
        if (!noError) {
            inputError.resize(n,d);
        }

        for (uint_t i = 0; i < n; ++i) {
            uint_t o = valid[i];
            output[i] = oldOutput[o];
            input.row(i) = oldInput.row(o);
            if (!noError) {
                inputError.row(i) = oldError.row(o);
            }
        }
    }
}

void GPz::splitTrainValid_(const Mat2d& input, const Mat2d& inputError,
    const Vec1d& output, const Vec1d& weight) {

    if (trainValidRatio_ == 1.0) {
        // No validation set
        inputTrain_ = input;
        inputErrorTrain_ = inputError;
        outputTrain_ = output;
        weightTrain_ = weight;

        inputValid_.resize(0,0);
        inputErrorValid_.resize(0,0);
        outputValid_.resize(0);
        weightValid_.resize(0);

        if (verbose_) {
            std::cout << "not using a validation set" << std::endl;
        }
    } else {
        std::vector<uint_t> indices(input.rows());
        std::iota(indices.begin(), indices.end(), 0u);

        if (trainValidSplitMethod_ == TrainValidationSplitMethod::RANDOM) {
            // Randomly shuffle the data
            std::mt19937 seed(seedTrainSplit_);
            std::shuffle(indices.begin(), indices.end(), seed);
        }

        uint_t numberTrain = round(input.rows()*trainValidRatio_);
        if (numberTrain == 0) {
            throw std::runtime_error("cannot have zero training data points");
        }

        uint_t numberValid = input.rows() - numberTrain;

        // Pick training data from first part of shuffled data
        inputTrain_.resize(numberTrain, numberFeatures_);
        if (inputError.rows() != 0) {
            inputErrorTrain_.resize(numberTrain, numberFeatures_);
        } else {
            inputErrorTrain_.resize(0,0);
        }
        outputTrain_.resize(numberTrain);
        weightTrain_.resize(numberTrain);

        for (uint_t i = 0; i < numberTrain; ++i) {
            uint_t index = indices[i];
            outputTrain_[i] = output[index];
            weightTrain_[i] = weight[index];
            for (uint_t j = 0; j < numberFeatures_; ++j) {
                inputTrain_(i,j) = input(index,j);
                if (inputError.rows() != 0) {
                    inputErrorTrain_(i,j) = inputError(index,j);
                }
            }
        }

        sumWeightTrain_ = weightTrain_.sum();

        // Pick validation data from second part of shuffled data
        inputValid_.resize(numberValid, numberFeatures_);
        if (inputError.rows() != 0) {
            inputErrorValid_.resize(numberValid, numberFeatures_);
        } else {
            inputErrorValid_.resize(0,0);
        }
        outputValid_.resize(numberValid);
        weightValid_.resize(numberValid);

        for (uint_t i = 0; i < numberValid; ++i) {
            uint_t index = indices[i + numberTrain];
            outputValid_[i] = output[index];
            weightValid_[i] = weight[index];
            for (uint_t j = 0; j < numberFeatures_; ++j) {
                inputValid_(i,j) = input(index,j);
                if (inputError.rows() != 0) {
                    inputErrorValid_(i,j) = inputError(index,j);
                }
            }
        }

        sumWeightValid_ = weightValid_.sum();

        if (verbose_) {
            std::cout << "split fitting data into " << inputTrain_.rows() << " for training and "
                << inputValid_.rows() << " for validation" << std::endl;
        }
    }
}

Vec1d GPz::computeWeights_(const Vec1d& output) const {
    Vec1d weight;

    switch (weightingScheme_) {
        case WeightingScheme::UNIFORM: {
            weight.resize(output.rows());
            weight.fill(1.0);
            break;
        }
        case WeightingScheme::ONE_OVER_ONE_PLUS_OUTPUT: {
            weight = 1.0/((1.0 + output)*(1.0 + output));
            break;
        }
        case WeightingScheme::BALANCED: {
            // Make bins
            double minValue = output.minCoeff();
            double maxValue = output.maxCoeff();
            uint_t numBins = ceil((maxValue - minValue)/balancedWeightingBinSize_);

            Vec1d bins(numBins+1);
            std::iota(begin(bins), end(bins), 0.0);
            bins = bins*balancedWeightingBinSize_ + minValue;

            // Compute histogram of counts in bins
            uint_t maxCount = 0;
            weight.setZero(output.rows());
            histogram(output, bins, [&](uint_t /*index*/, histogram_iterator begin, histogram_iterator end) {
                uint_t count = end - begin;
                for (histogram_iterator iter = begin; iter != end; ++iter) {
                    weight[*iter] = count;
                }

                if (count > maxCount) {
                    maxCount = count;
                }
            });

            // Set weight as ~ 1/count and normalize to maximum weight of one
            weight = maxCount/weight;

            break;
        }
    }

    for (uint_t i = 0; i < output.rows(); ++i) {
        if (!std::isfinite(weight[i])) {
            throw std::runtime_error("invalid weight found for output value");
        }
    }

    return weight;
}

void GPz::initializeInputs_(Mat2d input, Mat2d inputError, Vec1d output) {
    // Cleanup the sample
    eraseInvalidTrainData_(input, inputError, output);

    // Compute weights and split training/valid
    Vec1d weight = computeWeights_(output);
    splitTrainValid_(input, inputError, output, weight);

    // Setup missing cache
    buildMissingCache_(input);
    missingTrain_ = getBestMissingID_(inputTrain_);
    missingValid_ = getBestMissingID_(inputValid_);

    // Create arrays, matrices, etc.
    updateNumberParameters_();
    resizeArrays_();

    // Apply normalizations
    normalizeTrainingInputs_(inputTrain_, inputErrorTrain_, missingTrain_,
        outputTrain_, weightTrain_);

    applyInputNormalization_(inputValid_, inputErrorValid_);
    applyOutputNormalization_(inputValid_, missingValid_, outputValid_);
}

void GPz::computeTrainingPCA_() {
    const uint_t d = numberFeatures_;
    const uint_t n = inputTrain_.rows();

    // Compute mean of each feature (filtering out missing data)
    for (uint_t j = 0; j < d; ++j) {
        featurePCAMean_[j] = 0.0;
        uint_t count = 0;
        for (uint_t i = 0; i < n; ++i) {
            if (!std::isnan(inputTrain_(i,j))) {
                featurePCAMean_[j] += inputTrain_(i,j);
                ++count;
            }
        }

        featurePCAMean_[j] /= count;
    }

    // Compute corellation matrix (filtering out missing data)
    for (uint_t j = 0; j < d; ++j)
    for (uint_t k = j; k < d; ++k) {
        double sum = 0.0;
        uint_t count = 0;
        for (uint_t i = 0; i < n; ++i) {
            if (!std::isnan(inputTrain_(i,j)) && !std::isnan(inputTrain_(i,k))) {
                double dj = inputTrain_(i,j) - featurePCAMean_[j];
                double dk = inputTrain_(i,k) - featurePCAMean_[k];
                sum += dj*dk;
                ++count;
            }
        }

        featurePCASigma_(j,k) = featurePCASigma_(k,j) = sum*(n/double(count));
    }

    // Compute eigen-values and eigen-vectors
    Eigen::EigenSolver<Mat2d> solver(featurePCASigma_);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("could not get eigenvectors of PCA sigma matrix");
    }

    Mat1d eigenValues = solver.eigenvalues().real();
    Mat2d eigenVectors = solver.eigenvectors().real();

    eigenValues = (eigenValues/(n-1.0)).cwiseSqrt();
    featurePCABasisVectors_ = eigenValues.asDiagonal()*eigenVectors.transpose();

    featurePCASigma_ /= n;
}

void GPz::initializeBasisFunctions_() {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    // std::ifstream in("init_p_ml.txt");
    // parameters_.basisFunctionPositions.resize(m,d);
    // char comma;
    // for (uint_t i = 0; i < m; ++i)
    // for (uint_t j = 0; j < d; ++j) {
    //     if (!(in >> parameters_.basisFunctionPositions(i,j))) {
    //         std::cerr << "could not read position for " << i << "," << j << std::endl;
    //     }

    //     if (j != d-1) {
    //         in >> comma;
    //         if (comma != ',') {
    //             std::cerr << "unexpected character '" << comma << "'" << std::endl;
    //         }
    //     }
    // }

    std::mt19937 seed(seedPositions_);
    // Uniform distribution between -sqrt(3) and sqrt(3) has mean of zero and standard deviation of unity
    std::uniform_real_distribution<double> uniform(-sqrt(3.0), sqrt(3.0));

    // Populate the basis function positions with random numbers
    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j < d; ++j) {
        parameters_.basisFunctionPositions(i,j) = uniform(seed);
    }

    // Apply the PCA de-projection to mimic correlations of the data
    parameters_.basisFunctionPositions = parameters_.basisFunctionPositions*featurePCABasisVectors_;

    // Add data mean
    for (uint_t i = 0; i < m; ++i) {
        parameters_.basisFunctionPositions.row(i) += featurePCAMean_.matrix().transpose();
    }

    if (verbose_) {
        std::cout << "initialized BF positions" << std::endl;
        // for (uint_t i = 0; i < m; ++i) {
        //     std::cout << "  ";
        //     for (uint_t j = 0; j < d; ++j) {
        //         std::cout << parameters_.basisFunctionPositions(i,j) << ",";
        //     }
        //     std::cout << std::endl;
        // }
    }
}

void GPz::initializeBasisFunctionRelevances_() {
    const uint_t n = outputTrain_.rows();
    double outputLogVariance = log(outputTrain_.square().sum()/(n-1.0));
    parameters_.basisFunctionLogRelevances.fill(-outputLogVariance);

    if (verbose_) {
        std::cout << "initialized BF relevances" << std::endl;
        // std::cout << "  " << -outputLogVariance << std::endl;
    }
}

void GPz::buildMissingCache_(const Mat2d& input) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // Cache system to save data related to each combination of missing bands
    // encountered in the training data

    missingCache_.clear();
    noMissingCache_ = nullptr;

    // Iterate over input elements and fill in missing data
    for (uint_t i = 0; i < n; ++i) {
        std::vector<bool> missing(d);
        uint_t countMissing = 0;
        for (uint_t j = 0; j < d; ++j) {
            missing[j] = std::isnan(input(i,j));
            if (missing[j]) {
                ++countMissing;
            }
        }

        // See if this combination of missing bands has already been cached
        bool matchFound = false;
        for (auto& cacheItem : missingCache_) {
            bool match = true;
            for (uint_t j = 0; j < d; ++j) {
                if (missing[j] != cacheItem.missing[j]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                matchFound = true;
                break;
            }
        }

        if (matchFound) {
            // Already taken care of
            continue;
        }

        // Combination not found, add new cache entry
        MissingCacheElement newCache;
        newCache.id = missingCache_.size();
        newCache.countMissing = countMissing;
        newCache.missing = missing;

        newCache.indexMissing.resize(d);
        newCache.indexObserved.resize(d);
        for (uint_t j = 0, u = 0, o = 0; j < d; ++j) {
            if (missing[j]) {
                newCache.indexMissing[j] = u;
                newCache.indexObserved[j] = -1;
                ++u;
            } else {
                newCache.indexObserved[j] = o;
                newCache.indexMissing[j] = -1;
                ++o;
            }
        }

        missingCache_.push_back(newCache);
    }

    for (auto& e : missingCache_) {
        if (e.countMissing == 0) {
            noMissingCache_ = &e;
            break;
        }
    }

    assert(noMissingCache_ != nullptr && "bug: no missing cache with zero missing element");

    if (verbose_) {
        std::cout << "missing data cache built (" << missingCache_.size() << " combinations found)"
            << std::endl;
    }
}

const GPz::MissingCacheElement* GPz::findMissingCacheElement_(int id) const {
    // Fast method, assumes the cache is sorted by increasing ID
    // (true by construction as long as the implementation of buildMissingCacheTrain_() is not modified)

    if (id >= 0 && static_cast<uint_t>(id) < missingCache_.size()) {
        return &missingCache_[id];
    } else {
        return nullptr;
    }

    // Slow but more foolprof method (assumes no ordering)

    // auto iter = std::find_if(missingCache_.begin(), missingCache_.end(),
    //     [&](const MissingCacheElement& element) {
    //         return element.id == id;
    //     }
    // );

    // if (iter == missingCache_.end()) {
    //     return nullptr;
    // } else {
    //     return &*iter;
    // }
}

const GPz::MissingCacheElement& GPz::getMissingCacheElement_(int id) const {
    const MissingCacheElement* element = findMissingCacheElement_(id);
    assert(element != nullptr && "bug: missing cache element");
    return *element;
}

Vec1i GPz::getBestMissingID_(const Mat2d& input) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    Vec1i result(n);
    for (uint_t i = 0; i < n; ++i) {
        std::vector<bool> missing(d);
        for (uint_t j = 0; j < d; ++j) {
            missing[j] = std::isnan(input(i,j));
        }

        // See if this combination of missing bands exists in the cache
        int bestDistance = d+1;
        int bestID = -1;
        for (auto& cacheItem : missingCache_) {
            int distance = 0;
            bool match = true;
            for (uint_t j = 0; j < d; ++j) {
                if (missing[j] && !cacheItem.missing[j]) {
                    // This cache element is not missing a band but
                    // the galaxy is, discard
                    match = false;
                    break;
                }

                // For each mis-matching missing bands, add
                // one to the distance, so the cache element with
                // the closest distance is the best matching combination
                if (missing[j] != cacheItem.missing[j]) {
                    ++distance;
                }
            }

            if (match && distance < bestDistance) {
                bestDistance = distance;
                bestID = cacheItem.id;
            }
        }

        assert(bestID >= 0 && "bug: object not found in missing cache");

        result[i] = bestID;
    }

    return result;
}

void GPz::fetchMatrixElements_(Mat2d& out, const Mat2d& in, const MissingCacheElement& element,
    char first, char second) const {

    const uint_t d = numberFeatures_;

    // Early exit for no missing
    if (first == 'o' && second == 'o' && element.countMissing == 0) {
        out = in;
        return;
    }

    uint_t nfirst = 0;
    uint_t nsecond = 0;
    switch (first) {
        case ':': nfirst = d; break;
        case 'u': nfirst = element.countMissing; break;
        case 'o': nfirst = d - element.countMissing; break;
        default : assert(false && "should not happen"); break;
    }
    switch (second) {
        case ':': nsecond = d; break;
        case 'u': nsecond = element.countMissing; break;
        case 'o': nsecond = d - element.countMissing; break;
        default : assert(false && "should not happen"); break;
    }

    out.resize(nfirst, nsecond);

    // Early exit for no missing
    if ((first == 'u' || second == 'u') && element.countMissing == 0) {
        return;
    }

    for (uint_t j = 0, l = 0; j < d; ++j) {
        bool goodFirst = false;
        switch (first) {
            case ':': goodFirst = true; break;
            case 'u': goodFirst = element.missing[j] == true; break;
            case 'o': goodFirst = element.missing[j] == false; break;
            default : assert(false && "should not happen"); break;
        }

        if (goodFirst) {
            for (uint_t k = 0, q = 0; k < d; ++k) {
                bool goodSecond = false;
                switch (second) {
                    case ':': goodSecond = true; break;
                    case 'u': goodSecond = element.missing[k] == true; break;
                    case 'o': goodSecond = element.missing[k] == false; break;
                    default : assert(false && "should not happen"); break;
                }

                if (goodSecond) {
                    out(l,q) = in(j,k);
                    ++q;
                }
            }

            ++l;
        }
    }
}

void GPz::fetchVectorElements_(Mat1d& out, const Mat1d& in, const MissingCacheElement& element,
    char first) const {

    const uint_t d = numberFeatures_;

    // Early exit for no missing
    if (first == 'o' && element.countMissing == 0) {
        out = in;
        return;
    }

    uint_t nfirst = 0;
    switch (first) {
        case ':': nfirst = d; break;
        case 'u': nfirst = element.countMissing; break;
        case 'o': nfirst = d - element.countMissing; break;
        default : assert(false && "should not happen"); break;
    }

    out.resize(nfirst);

    // Early exit for no missing
    if (first == 'u' && element.countMissing == 0) {
        return;
    }

    for (uint_t j = 0, l = 0; j < d; ++j) {
        bool goodFirst = false;
        switch (first) {
            case ':': goodFirst = true; break;
            case 'u': goodFirst = element.missing[j] == true; break;
            case 'o': goodFirst = element.missing[j] == false; break;
            default : assert(false && "should not happen"); break;
        }

        if (goodFirst) {
            out[l] = in[j];
            ++l;
        }
    }
}

void GPz::addMatrixElements_(Mat2d& out, const Mat2d& in, const MissingCacheElement& element,
    char first, char second) const {

    const uint_t d = numberFeatures_;

    // Early exit for no missing
    if (first == 'o' && second == 'o' && element.countMissing == 0) {
        out += in;
        return;
    }
    if ((first == 'u' || second == 'u') && element.countMissing == 0) {
        return;
    }

    for (uint_t j = 0, l = 0; j < d; ++j) {
        bool goodFirst = false;
        switch (first) {
            case ':': goodFirst = true; break;
            case 'u': goodFirst = element.missing[j] == true; break;
            case 'o': goodFirst = element.missing[j] == false; break;
            default : assert(false && "should not happen"); break;
        }

        if (goodFirst) {
            for (uint_t k = 0, q = 0; k < d; ++k) {
                bool goodSecond = false;
                switch (second) {
                    case ':': goodSecond = true; break;
                    case 'u': goodSecond = element.missing[k] == true; break;
                    case 'o': goodSecond = element.missing[k] == false; break;
                    default : assert(false && "should not happen"); break;
                }

                if (goodSecond) {
                    out(j,k) += in(l,q);
                    ++q;
                }
            }

            ++l;
        }
    }
}

void GPz::buildLinearPredictorCache_() {
    // Iterate over cache entries and build linear predictor matrix
    for (auto& cacheItem : missingCache_) {
        // Extract sub-blocks of the PCASigma matrix
        // sigma(observed,missing) and sigma(observed,observed)
        Mat2d sigmaMissing, sigmaObserved;
        fetchMatrixElements_(sigmaMissing,  featurePCASigma_, cacheItem, 'o', 'u');
        fetchMatrixElements_(sigmaObserved, featurePCASigma_, cacheItem, 'o', 'o');

        // Compute Cholesky decomposition of sigma(observed,observed)
        // and compute predictor by solving for sigma(observed,missing)
        Eigen::LDLT<Mat2d> cholesky(sigmaObserved);
        cacheItem.predictor = cholesky.solve(sigmaMissing);
    }
}

Mat2d GPz::initializeCovariancesFillLinear_(Mat2d input, const Vec1i& missing) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // Iterate over input elements and fill in missing data
    for (uint_t i = 0; i < n; ++i) {
        const MissingCacheElement& cache = getMissingCacheElement_(missing[i]);

        if (cache.countMissing == 0) continue;

        // Consolidate observed bands in one array
        Mat1d observed(d - cache.countMissing);
        for (uint_t j = 0, k = 0; j < d; ++j) {
            if (!cache.missing[j]) {
                observed[k] = input(i,j) - featurePCAMean_[j];
                ++k;
            }
        }

        // Fill in missing data
        Mat1d missingFilled = observed.transpose()*cache.predictor;
        for (uint_t j = 0, k = 0; j < d; ++j) {
            if (cache.missing[j] && std::isnan(input(i,j))) {
                input(i,j) = missingFilled[k] + featurePCAMean_[j];
                ++k;
            }
        }
    }

    return input;
}

Vec1d GPz::initializeCovariancesMakeGamma_(const Mat2d& input, const Vec1i& missing) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // Replace missing data by linear predictions based on observed data
    Mat2d linearInputs = initializeCovariancesFillLinear_(input, missing);

    Vec1d gamma(m);
    double factor = 0.5*pow(m, 1.0/d);
    for (uint_t i = 0; i < m; ++i) {
        double meanSquaredDist = 0.0;
        for (uint_t j = 0; j < n; ++j)
        for (uint_t k = 0; k < d; ++k) {
            double d = parameters_.basisFunctionPositions(i,k) - linearInputs(j,k);
            meanSquaredDist += d*d;
        }

        meanSquaredDist /= n;

        gamma[i] = sqrt(factor/meanSquaredDist);
    }

    return gamma;
}

void GPz::initializeCovariances_() {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    // Compute some statistics from training set
    Vec1d gamma = initializeCovariancesMakeGamma_(inputTrain_, missingTrain_);

    switch (covarianceType_) {
        case CovarianceType::GLOBAL_LENGTH:
        case CovarianceType::GLOBAL_DIAGONAL:
        case CovarianceType::GLOBAL_COVARIANCE: {
            double mean_gamma = gamma.mean();

            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = 0; k < d; ++k) {
                parameters_.basisFunctionCovariances[i](j,k) = (j == k ? mean_gamma : 0.0);
            }

            break;
        }
        case CovarianceType::VARIABLE_LENGTH:
        case CovarianceType::VARIABLE_DIAGONAL:
        case CovarianceType::VARIABLE_COVARIANCE: {
            for (uint_t i = 0; i < m; ++i)
            for (uint_t j = 0; j < d; ++j)
            for (uint_t k = 0; k < d; ++k) {
                parameters_.basisFunctionCovariances[i](j,k) = (j == k ? gamma[i] : 0.0);
            }

            break;
        }
    }

    if (verbose_) {
        std::cout << "initialized BF covariances" << std::endl;
        // std::cout << "  ";
        // for (uint_t i = 0; i < m; ++i) {
        //     std::cout << gamma[i] << ",";
        // }
        // std::cout << std::endl;
    }
}

void GPz::initializeErrors_() {
    const uint_t n = inputTrain_.rows();

    // Initialize constant term from variance of outputs
    double outputLogVariance = log(outputTrain_.square().sum()/(n-1.0));
    parameters_.logUncertaintyConstant = outputLogVariance;

    // Initialize basis function weights to zero
    // (only optimized for OutputUncertaintyType::INPUT_DEPENDENT)
    parameters_.uncertaintyBasisWeights.fill(0.0);
    parameters_.uncertaintyBasisLogRelevances.fill(0.0);

    if (verbose_) {
        std::cout << "initialized output errors" << std::endl;
        // std::cout << "  " << outputLogVariance << std::endl;
    }
}

void GPz::initializeFit_() {
    // Pre-compute some things
    computeTrainingPCA_();
    buildLinearPredictorCache_();

    // Set initial values for hyper-parameters
    initializeBasisFunctions_();
    initializeBasisFunctionRelevances_();
    initializeCovariances_();
    initializeErrors_();

    // TODO: we could free the memory used by the predictor cache here,
    // if it is not used by anything else but to initialize the covariances.

    if (verbose_) {
        std::cout << "initialized all fit parameters" << std::endl;
    }
}

bool GPz::checkInputDimensions_(const Mat2d& input) const {
    return static_cast<uint_t>(input.cols()) == numberFeatures_;
}

bool GPz::checkErrorDimensions_(const Mat2d& input, const Mat2d& inputError) const {
    bool noError = inputError.size() == 0;
    bool errorSameSize = inputError.rows() == input.rows() && inputError.cols() == input.cols();
    return noError || errorSameSize;
}

// =======================
// Internal functions: fit
// =======================

void GPz::updateMissingCache_(MissingCacheUpdate what) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;
    const bool diagonalCovariance = covarianceType_ == CovarianceType::VARIABLE_DIAGONAL ||
                                    covarianceType_ == CovarianceType::GLOBAL_DIAGONAL;

    std::vector<Mat2d> isigma(m);
    std::vector<Mat2d> sigma(m);
    for (uint_t i = 0; i < m; ++i) {
        const Mat2d& fullGamma = parameters_.basisFunctionCovariances[i];
        isigma[i] = fullGamma.transpose()*fullGamma;
        sigma[i] = computeInverseSymmetric(isigma[i]);
    }

    for (auto& cacheItem : missingCache_) {
        cacheItem.covariancesObserved.resize(m);
        cacheItem.covariancesMissing.resize(m);
        cacheItem.invCovariancesObserved.resize(m);
        cacheItem.covariancesObservedLogDeterminant.setZero(m);
        if (what == MissingCacheUpdate::TRAIN) {
            cacheItem.gUO.resize(m);
            cacheItem.dgO.resize(m);
        } else if (what == MissingCacheUpdate::PREDICT) {
            cacheItem.R.resize(m);
            cacheItem.T.resize(m);
            cacheItem.Psi_hat.resize(m);

            if (diagonalCovariance && optimizations_.specializeForDiagCovariance) {
                cacheItem.Nij.resize(m,m);
                cacheItem.Nu.resize(m);
                for (uint_t i = 0; i < m; ++i) {
                    cacheItem.Nu[i].resize(i+1);
                    for (uint_t j = 0; j <= i; ++j) {
                        cacheItem.Nu[i][j].resize(m);
                    }
                }
            }
        }

        for (uint_t i = 0; i < m; ++i) {
            const Mat2d& fullGamma = parameters_.basisFunctionCovariances[i]; // GPzMatLab: Gamma(:,:,i)

            if (cacheItem.countMissing < d) {
                // Fetch elements of the covariance matrix for non-missing bands
                fetchMatrixElements_(cacheItem.covariancesObserved[i], sigma[i], cacheItem, 'o', 'o');

                // Compute inverse
                cacheItem.invCovariancesObserved[i] = computeInverseSymmetric(cacheItem.covariancesObserved[i]);

                // Compute log determinant
                cacheItem.covariancesObservedLogDeterminant[i] = computeLogDeterminant(cacheItem.covariancesObserved[i]);
            } else {
                // All bands missing
                cacheItem.covariancesObserved[i].setZero(0,0);
                cacheItem.invCovariancesObserved[i].setZero(0,0);
                cacheItem.covariancesObservedLogDeterminant[i] = 0.0;
            }

            if (what == MissingCacheUpdate::TRAIN) {
                if (cacheItem.countMissing != 0) {
                    // Compute gUO and dgO (for training)
                    Mat2d gammaObserved; // GPzMatLab: Gamma(:,o,i)
                    fetchMatrixElements_(gammaObserved, fullGamma, cacheItem, ':', 'o');

                    if (diagonalCovariance && optimizations_.specializeForDiagCovariance) {
                        cacheItem.gUO[i].setZero(cacheItem.countMissing,d-cacheItem.countMissing);
                        cacheItem.dgO[i] = 2*gammaObserved;
                    } else {
                        Mat2d isigmaMissing;  // GPzMatLab: iSigma(u,u)
                        Mat2d isigmaMixed; // GPzMatLab: iSigma(u,o)
                        fetchMatrixElements_(isigmaMissing, isigma[i], cacheItem, 'u', 'u');
                        fetchMatrixElements_(isigmaMixed,   isigma[i], cacheItem, 'u', 'o');

                        Mat2d gammaMissing;  // GPzMatLab: Gamma(:,u,i)
                        fetchMatrixElements_(gammaMissing,  fullGamma, cacheItem, ':', 'u');

                        cacheItem.gUO[i] = computeInverseSymmetric(isigmaMissing)*isigmaMixed;
                        cacheItem.dgO[i] = 2*(gammaObserved - gammaMissing*cacheItem.gUO[i]);
                    }
                } else {
                    cacheItem.gUO[i].setZero(0,0);
                    cacheItem.dgO[i] = 2*fullGamma;
                }
            }

            if (what == MissingCacheUpdate::PREDICT) {
                fetchMatrixElements_(cacheItem.covariancesMissing[i], sigma[i], cacheItem, 'u', 'u');

                // Compute R and Psi_hat (for prediction)
                if (cacheItem.countMissing == 0) {
                    // No missing data, these variables are not used
                    cacheItem.Psi_hat[i].resize(0,0);
                    cacheItem.T[i].resize(0,0);
                    cacheItem.R[i].resize(0,0);
                } else if (cacheItem.countMissing == d) {
                    // All data missing, special case
                    cacheItem.Psi_hat[i] = sigma[i];
                    cacheItem.T[i].setZero(d,0);
                    cacheItem.R[i].setZero(0,d);
                } else {
                    // Just some data missing
                    Mat2d sigmaMissing;  // GPzMatLab: Sigma(u,u)
                    Mat2d sigmaMixed; // GPzMatLab: Sigma(o,u)
                    fetchMatrixElements_(sigmaMissing,  sigma[i], cacheItem, 'u', 'u');
                    fetchMatrixElements_(sigmaMixed,    sigma[i], cacheItem, 'o', 'u');

                    if (diagonalCovariance && optimizations_.specializeForDiagCovariance) {
                        cacheItem.R[i].setZero(d-cacheItem.countMissing,cacheItem.countMissing);
                        cacheItem.Psi_hat[i] = sigmaMissing;
                    } else {
                        Mat2d sigmaObserved; // GPzMatLab: Sigma(o,o)
                        fetchMatrixElements_(sigmaObserved, sigma[i], cacheItem, 'o', 'o');

                        Eigen::LDLT<Mat2d> cholesky(sigmaObserved);
                        cacheItem.R[i] = cholesky.solve(sigmaMixed);
                        cacheItem.Psi_hat[i] = sigmaMissing - sigmaMixed.transpose()*cacheItem.R[i];
                    }

                    // Original MatLab code
                    // uint_t no = d - cacheItem.countMissing;
                    // uint_t nu = cacheItem.countMissing;
                    // cacheItem.T[i].setZero(d, no);
                    // cacheItem.T[i].block(0,  0, no, no) = Mat2d::Identity(no, no);
                    // cacheItem.T[i].block(no, 0, nu, no) = cacheItem.R[i].transpose();

                    // What I think is correct
                    cacheItem.T[i].setZero(d, d - cacheItem.countMissing);
                    for (uint_t k = 0; k < d; ++k) {
                        if (cacheItem.missing[k]) {
                            cacheItem.T[i].row(k) = cacheItem.R[i].col(cacheItem.indexMissing[k]);
                        } else {
                            cacheItem.T[i](k,cacheItem.indexObserved[k]) = 1.0;
                        }
                    }
                }
            }
        }

        if (what == MissingCacheUpdate::PREDICT) {
            if (diagonalCovariance && optimizations_.specializeForDiagCovariance) {
                if (cacheItem.countMissing == 0) {
                    // No missing data, these variables are not used
                    cacheItem.Nij.fill(0.0);
                    for (uint_t i = 0; i < m; ++i)
                    for (uint_t j = 0; j <= i; ++j) {
                        cacheItem.Nu[i][j].fill(1.0);
                    }
                } else {
                    for (uint_t i = 0; i < m; ++i)
                    for (uint_t j = 0; j < m; ++j) {
                        if (j < i) {
                            cacheItem.Nij(i,j) = cacheItem.Nij(j,i);
                        } else {
                            // Compute Nij
                            // TODO: this can be specialized for diag covariance directly
                            Mat2d Sij = cacheItem.covariancesMissing[i]
                                      + cacheItem.covariancesMissing[j];

                            Eigen::JacobiSVD<Mat2d> svd(Sij, Eigen::ComputeThinU | Eigen::ComputeThinV);

                            Mat1d Delta = parameters_.basisFunctionPositions.row(i)
                                        - parameters_.basisFunctionPositions.row(j);

                            Mat1d DeltaMissing;
                            fetchVectorElements_(DeltaMissing, Delta, cacheItem, 'u');

                            double lnNij = svd.solve(DeltaMissing).transpose()*DeltaMissing
                                         + computeLogDeterminant(svd);

                            cacheItem.Nij(i,j) = exp(-0.5*lnNij);

                            // Compute Nu
                            for (uint_t l = 0; l < m; ++l) {
                                double lnN = 0.0;
                                double det = 1.0;
                                for (uint_t k = 0; k < d; ++k) {
                                    if (cacheItem.missing[k]) {
                                        double icovi = isigma[i](k,k);
                                        double icovj = isigma[j](k,k);
                                        double posi = parameters_.basisFunctionPositions(i,k);
                                        double posj = parameters_.basisFunctionPositions(j,k);
                                        double Cij = 1.0/(icovi + icovj);
                                        double cij = (posi*icovi + posj*icovj)*Cij;
                                        double Delta = parameters_.basisFunctionPositions(l,k) - cij;
                                        Cij += sigma[l](k,k);
                                        lnN += square(Delta)/Cij;
                                        det *= Cij;
                                    }
                                }

                                cacheItem.Nu[j][i][l] = exp(-0.5*lnN)/sqrt(det);
                            }
                        }
                    }
                }
            }
        }
    }
}

void GPz::updateTrainBasisFunctions_() {
    updateBasisFunctions_(trainBasisFunctions_, inputTrain_, inputErrorTrain_, missingTrain_);
}

void GPz::updateValidBasisFunctions_() {
    updateBasisFunctions_(validBasisFunctions_, inputValid_, inputErrorValid_, missingValid_);
}

void GPz::updateTrainOutputErrors_() {
    trainOutputLogError_ = evaluateOutputLogErrors_(trainBasisFunctions_);
}

void GPz::updateValidOutputErrors_() {
    validOutputLogError_ = evaluateOutputLogErrors_(validBasisFunctions_);
}

Mat1d GPz::evaluateBasisFunctionsGeneral_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    Eigen::JacobiSVD<Mat2d> svd;

    Mat1d deltaAll;
    Mat1d delta; // GPzMatLab: Delta
    Mat1d deltaSolved; // GPzMatLab: Delta/Sigma(o,o) or Delta/PsiPlusSigma
    Mat1d varianceObserved;
    Mat2d covariance; // GPzMatLab: PsiPlusSigma

    const double log2 = log(2.0);

    Mat1d funcs(m);
    for (uint_t j = 0; j < m; ++j) {
        double value = log2*element.countMissing;

        if (element.countMissing < d) {
            deltaAll = input - parameters_.basisFunctionPositions.row(j).transpose();
            fetchVectorElements_(delta, deltaAll, element, 'o');

            if (inputError.rows() == 0) {
                deltaSolved = element.invCovariancesObserved[j]*delta;
            } else {
                fetchVectorElements_(varianceObserved, inputError, element, 'o');

                covariance = element.covariancesObserved[j];
                covariance.diagonal() += varianceObserved;

                svd.compute(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
                deltaSolved = svd.solve(delta);
                double ldet = computeLogDeterminant(svd);
                double ldetobs = element.covariancesObservedLogDeterminant[j];

                value += ldet - ldetobs;
            }

            value += (delta.array()*deltaSolved.array()).sum();
        }

        funcs[j] = exp(-0.5*value);
    }

    return funcs;
}

Mat1d GPz::evaluateBasisFunctionsDiag_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    const double log2 = log(2.0);

    Mat1d funcs(m);

    // Specialization of code for one single feature or diagonal covariances (faster)
    if (inputError.rows() == 0) {
        for (uint_t j = 0; j < m; ++j) {
            double value = log2*element.countMissing;

            for (uint_t k = 0; k < d; ++k) {
                if (!element.missing[k]) {
                    int o = element.indexObserved[k];
                    double delta = input[k] - parameters_.basisFunctionPositions(j,k);
                    value += square(delta)*element.invCovariancesObserved[j](o,o);
                }
            }

            funcs[j] = exp(-0.5*value);
        }
    } else {
        for (uint_t j = 0; j < m; ++j) {
            double value = log2*element.countMissing;

            if (d == 1) {
                if (element.countMissing == 1) {
                    funcs[j] = exp(-0.5*value);
                } else {
                    double delta = input[0] - parameters_.basisFunctionPositions(j,0);
                    double covariance = element.covariancesObserved[j](0,0) + inputError[0]; // GPzMatLab: PsiPlusSigma
                    value += square(delta)/covariance;
                    funcs[j] = exp(-0.5*value)/sqrt(1.0 + inputError[0]/element.covariancesObserved[j](0,0));
                }
            } else {
                double det = 1.0;
                for (uint_t k = 0; k < d; ++k) {
                    if (!element.missing[k]) {
                        int o = element.indexObserved[k];
                        double delta = input[k] - parameters_.basisFunctionPositions(j,k);
                        double covariance = element.covariancesObserved[j](o,o) + inputError[k]; // GPzMatLab: PsiPlusSigma
                        value += square(delta)/covariance;
                        det *= covariance;
                    }
                }

                double ldetobs = element.covariancesObservedLogDeterminant[j];
                value += log(det) - ldetobs;

                funcs[j] = exp(-0.5*value);
            }
        }
    }

    return funcs;
}

Mat1d GPz::evaluateBasisFunctions_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const {
    const uint_t d = numberFeatures_;

    const bool diagonalCovariance = covarianceType_ == CovarianceType::VARIABLE_DIAGONAL ||
                                    covarianceType_ == CovarianceType::GLOBAL_DIAGONAL;

    if ((d == 1 && optimizations_.specializeForSingleFeature) ||
        (diagonalCovariance == optimizations_.specializeForDiagCovariance)) {
        return evaluateBasisFunctionsDiag_(input, inputError, element);
    } else {
        return evaluateBasisFunctionsGeneral_(input, inputError, element);
    }
}

void GPz::updateBasisFunctions_(Mat2d& funcs, const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const {
    const uint_t n = input.rows();
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    if (funcs.rows() == 0) {
        funcs.resize(n,m);
    }

    const double log2 = log(2.0);

    const bool diagonalCovariance = covarianceType_ == CovarianceType::VARIABLE_DIAGONAL ||
                                    covarianceType_ == CovarianceType::GLOBAL_DIAGONAL;

    auto computeForSource = [&](uint_t i) {
        const MissingCacheElement& element = getMissingCacheElement_(missing[i]);

        if ((d == 1 && optimizations_.specializeForSingleFeature) ||
            (diagonalCovariance == optimizations_.specializeForDiagCovariance)) {

            // Specialization of code for one single feature or diagonal covariances (faster)
            if (inputError.rows() == 0) {
                for (uint_t j = 0; j < m; ++j) {
                    double value = log2*element.countMissing;

                    for (uint_t k = 0; k < d; ++k) {
                        if (!element.missing[k]) {
                            int o = element.indexObserved[k];
                            double delta = input(i,k) - parameters_.basisFunctionPositions(j,k);
                            value += square(delta)*element.invCovariancesObserved[j](o,o);
                        }
                    }

                    funcs(i,j) = exp(-0.5*value);
                }
            } else {
                for (uint_t j = 0; j < m; ++j) {
                    double value = log2*element.countMissing;

                    if (d == 1) {
                        if (element.countMissing == 1) {
                            funcs(i,j) = exp(-0.5*value);
                        } else {
                            double delta = input(i,0) - parameters_.basisFunctionPositions(j,0);
                            double covariance = element.covariancesObserved[j](0,0) + inputError(i,0); // GPzMatLab: PsiPlusSigma
                            value += square(delta)/covariance;
                            funcs(i,j) = exp(-0.5*value)/sqrt(1.0 + inputError(i,0)/element.covariancesObserved[j](0,0));
                        }
                    } else {
                        double det = 1.0;
                        for (uint_t k = 0; k < d; ++k) {
                            if (!element.missing[k]) {
                                int o = element.indexObserved[k];
                                double delta = input(i,k) - parameters_.basisFunctionPositions(j,k);
                                double covariance = element.covariancesObserved[j](o,o) + inputError(i,k); // GPzMatLab: PsiPlusSigma
                                value += square(delta)/covariance;
                                det *= covariance;
                            }
                        }

                        double ldetobs = element.covariancesObservedLogDeterminant[j];
                        value += log(det) - ldetobs;

                        funcs(i,j) = exp(-0.5*value);
                    }
                }
            }
        } else {
            // General code for any number of features
            if (inputError.rows() == 0) {
                funcs.row(i) = evaluateBasisFunctionsGeneral_(input.row(i), Mat1d{}, element).transpose();
            } else {
                funcs.row(i) = evaluateBasisFunctionsGeneral_(input.row(i), inputError.row(i), element).transpose();
            }
        }
    };

    parallel_for pool(optimizations_.enableMultithreading ? optimizations_.maxThreads : 0);
    pool.execute(computeForSource, n);
}

Mat2d GPz::evaluateBasisFunctions_(const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const {
    Mat2d funcs;
    updateBasisFunctions_(funcs, input, inputError, missing);
    return funcs;
}

double GPz::evaluateOutputLogError_(const Mat1d& basisFunctions) const {
    // Constant term
    double error = parameters_.logUncertaintyConstant;

    if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
        // Input-dependent parametrization using basis functions
        error += basisFunctions.transpose()*parameters_.uncertaintyBasisWeights;
    }

    return error;
}

Mat1d GPz::evaluateOutputLogErrors_(const Mat2d& basisFunctions) const {
    const uint_t n = basisFunctions.rows();

    Mat1d errors(n);

    // Constant term
    errors.fill(parameters_.logUncertaintyConstant);

    if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
        // Input-dependent parametrization using basis functions
        errors += basisFunctions*parameters_.uncertaintyBasisWeights;
    }

    return errors;
}

void GPz::updateTrainModel_(Minimize::FunctionOutput requested) {
    const uint_t n = inputTrain_.rows();
    const uint_t d = numberFeatures_;
    const uint_t m = numberBasisFunctions_;

    const bool updateLikelihood =
        requested == Minimize::FunctionOutput::ALL_TRAIN ||
        requested == Minimize::FunctionOutput::METRIC_TRAIN;

    const bool updateDerivatives =
        requested == Minimize::FunctionOutput::ALL_TRAIN ||
        requested == Minimize::FunctionOutput::DERIVATIVES_TRAIN;

    const bool diagonalCovariance = covarianceType_ == CovarianceType::VARIABLE_DIAGONAL ||
                                    covarianceType_ == CovarianceType::GLOBAL_DIAGONAL;

    if (updateLikelihood) {
        logLikelihood_ = 0.0;
    }

    // Pre-compute things
    double timePrev = 0, timeNow = 0;
    if (profileTraining_) {
        std::cout << "#### " << (int)requested << std::endl;
        timePrev = now();
    }

    updateMissingCache_(MissingCacheUpdate::TRAIN);

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateMissingCache " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    updateTrainBasisFunctions_();

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateTrainBasisFunctions " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    updateTrainOutputErrors_();

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateTrainOutputErrors " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    // Do the hard work...
    Mat1d relevances = parameters_.basisFunctionLogRelevances.array().exp().matrix(); // GPzMatLab: alpha
    Mat1d errorRelevances;     // GPzMatLab: tau
    Mat1d errorWeightsSquared; // GPzMatLab: v.^2
    if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
        errorRelevances = parameters_.uncertaintyBasisLogRelevances.array().exp().matrix();
        errorWeightsSquared = parameters_.uncertaintyBasisWeights.array().square();
    }

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateLikelihood 1 " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    Vec1d trainOutputError = (-trainOutputLogError_).array().exp(); // GPzMatLab: beta
    Vec1d dataWeight = weightTrain_*trainOutputError; // GPzMatLab: omega_x_beta
    Mat2d weightedBasisFunctions = trainBasisFunctions_; // GPzMatLab: BxPHI
    for (uint_t i = 0; i < n; ++i) {
        weightedBasisFunctions.row(i) *= dataWeight[i];
    }

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateLikelihood 2 " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    Mat2d modelCovariance = weightedBasisFunctions.transpose()*trainBasisFunctions_;
    modelCovariance.diagonal() += relevances; // GPzMatLab: SIGMA

    Eigen::LDLT<Mat2d> chol(modelCovariance);
    modelInvCovariance_ = computeInverseSymmetric(modelCovariance, chol);

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateLikelihood 3 " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    parallel_for pool(optimizations_.enableMultithreading ? optimizations_.maxThreads : 0);

    Mat2d solvedBasisFunctions(m,n);
    Mat2d solvedWeightedBasisFunctions(m,n);
    auto solveBasis = [&](uint_t i) {
        solvedBasisFunctions.col(i) = chol.solve(trainBasisFunctions_.row(i).transpose());
        solvedWeightedBasisFunctions.col(i) = solvedBasisFunctions.col(i)*dataWeight[i];
    };

    pool.execute(solveBasis, n);

    modelWeights_ = solvedWeightedBasisFunctions*outputTrain_.matrix();

    Mat1d deviates = trainBasisFunctions_*modelWeights_ - outputTrain_.matrix(); // GPzMatLab: delta
    Mat1d weightedDeviates = (dataWeight.array()*deviates.array()).matrix(); // GpzMatLab: omega_beta_x_delta

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateLikelihood 4 " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    if (updateLikelihood) {
        // Log likelihood
        // ==============

        double a1 = -0.5*(weightedDeviates.array()*deviates.array()).sum();
        double a2 = -0.5*(relevances.array()*modelWeights_.array().square()).sum();
        double a3 = +0.5*parameters_.basisFunctionLogRelevances.sum();
        double a4 = -0.5*computeLogDeterminant(chol);
        double a5 = +0.5*((-trainOutputLogError_).array()*weightTrain_.array()).sum();
        double a6 = -0.5*log(2.0*M_PI)*sumWeightTrain_;

        double a7 = 0.0, a8 = 0.0, a9 = 0.0;
        if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
            a7 = -0.5*(errorWeightsSquared.array()*errorRelevances.array()).sum();
            a8 = +0.5*parameters_.uncertaintyBasisLogRelevances.sum();
            a9 = -0.5*m*log(2.0*M_PI);
        }

        logLikelihood_ = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
    }

    if (profileTraining_) {
        timeNow = now();
        std::cout << "updateLikelihood 5 " << timeNow - timePrev << std::endl;
        timePrev = timeNow;
    }

    if (updateDerivatives) {
        // Derivative of basis functions
        // =============================

        Mat2d derivBasis = -solvedWeightedBasisFunctions.transpose()
            -weightedDeviates*modelWeights_.transpose(); // GPzMatLab: dlnPHI

        // Derivative wrt relevance
        // ========================

        Mat1d dwda = -chol.solve((relevances.array()*modelWeights_.array()).matrix());

        derivatives_.basisFunctionLogRelevances = 0.5
            -0.5*modelInvCovariance_.diagonal().array()*relevances.array()
            -(trainBasisFunctions_.transpose()*weightedDeviates).array()*dwda.array()
            -relevances.array()*modelWeights_.array()*dwda.array()
            -0.5*relevances.array()*modelWeights_.array().square();

        if (profileTraining_) {
            timeNow = now();
            std::cout << "updateDerivatives 0 " << timeNow - timePrev << std::endl;
            timePrev = timeNow;
        }

        // Derivative wrt uncertainty constant
        // ===================================

        Mat1d nu = (solvedBasisFunctions.transpose().array()*trainBasisFunctions_.array())
           .rowwise().sum(); // GPzMatLab: nu

        Mat1d derivOutputError(n); // GPzMatLab: dbeta
        // Do this in an explicit loop as all operations are component-wise, it's clearer
        for (uint_t i = 0; i < n; ++i) {
            derivOutputError[i] = -0.5*weightTrain_[i]*
                (1.0 - trainOutputError[i]*(deviates[i]*deviates[i] + nu[i]));
        }

        derivatives_.logUncertaintyConstant = derivOutputError.sum();

        if (profileTraining_) {
            timeNow = now();
            std::cout << "updateDerivatives 1 " << timeNow - timePrev << std::endl;
            timePrev = timeNow;
        }

        if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
            // Derivative wrt uncertainty weights
            // ==================================

            Mat1d weightedRelevance = parameters_.uncertaintyBasisWeights.array()*errorRelevances.array();

            derivatives_.uncertaintyBasisWeights = trainBasisFunctions_.transpose()*derivOutputError
                -weightedRelevance;

            // Derivative wrt uncertainty relevances
            // =====================================

            derivatives_.uncertaintyBasisLogRelevances = 0.5
                -0.5*weightedRelevance.array()*errorRelevances.array();

            // Contribution to derivative of basis functions
            // =============================================

            derivBasis += derivOutputError*parameters_.uncertaintyBasisWeights.transpose();
        }

        // Derivatives wrt to basis positions & covariances
        // ================================================

        derivBasis = derivBasis.array()*trainBasisFunctions_.array(); // GPzMatLab: dPHI

        derivatives_.basisFunctionPositions.fill(0.0);
        for (uint_t i = 0; i < m; ++i) {
            derivatives_.basisFunctionCovariances[i].fill(0.0);
        }

        if (profileTraining_) {
            timeNow = now();
            std::cout << "updateDerivatives 2 " << timeNow - timePrev << std::endl;
            timePrev = timeNow;
        }

        // Generic code for any number of feature and any covariance type
        auto derivativeAddContribution = [&](uint_t j) {
            Eigen::LDLT<Mat2d> chol;
            Mat2d derivInvCovariance; // GPzMatLab: diSoo
            Mat2d covariance, derivCovariance;
            Mat1d variance; // GPzMatLab: Psi(:,:,i) (only diagonal)
            Mat1d varianceObserved; // GPzMatLab: Psi(o,o,i) (only diagonal)
            Mat2d dgO, dgU;
            Mat2d covObsInvL;
            Mat1d deltaAll;
            Mat1d delta; // GPzMatLab: Delta
            Mat1d deltaSolved; // GPzMatLab: Delta/Sigma(o,o) or Delta/iPSoo
            Mat1d deltaSolvedCov;

            for (uint_t i = 0; i < n; ++i) {
                const MissingCacheElement& element = getMissingCacheElement_(missingTrain_[i]);

                deltaAll = inputTrain_.row(i) - parameters_.basisFunctionPositions.row(j);
                fetchVectorElements_(delta, deltaAll, element, 'o');

                if (inputErrorTrain_.rows() == 0) {
                    // Case with no input error
                    deltaSolved = element.invCovariancesObserved[j]*delta;
                    derivInvCovariance = (-0.5*derivBasis(i,j))*delta*delta.transpose();
                } else {
                    // Generic code for any covariance type
                    variance = inputErrorTrain_.row(i);
                    fetchVectorElements_(varianceObserved, variance, element, 'o');

                    covariance = element.covariancesObserved[j];
                    covariance.diagonal() += varianceObserved;

                    chol.compute(covariance);
                    deltaSolved = chol.solve(delta);

                    deltaSolvedCov = element.covariancesObserved[j]*deltaSolved;
                    covObsInvL = chol.solve(element.covariancesObserved[j]);
                    derivInvCovariance = (-0.5*derivBasis(i,j))*(
                        element.covariancesObserved[j]
                        - element.covariancesObserved[j]*covObsInvL.transpose()
                        + deltaSolvedCov*deltaSolvedCov.transpose()
                    );
                }

                // Derivative wrt to basis positions
                // =================================
                for (uint_t k = 0; k < d; ++k) {
                    if (!element.missing[k]) {
                        uint_t o = element.indexObserved[k];
                        derivatives_.basisFunctionPositions(j,k) += derivBasis(i,j)*deltaSolved[o];
                    }
                }

                // Derivative wrt to basis covariances
                // =================================
                dgO = element.dgO[j]*derivInvCovariance;
                addMatrixElements_(derivatives_.basisFunctionCovariances[j], dgO, element, ':', 'o');

                if (element.countMissing != 0) {
                    dgU = -dgO*element.gUO[j].transpose();
                    addMatrixElements_(derivatives_.basisFunctionCovariances[j], dgU, element, ':', 'u');
                }
            }
        };

        // Specialization for one single feature or diagonal covariance (faster)
        auto derivativeAddContributionDiag = [&](uint_t j) {
            for (uint_t i = 0; i < n; ++i) {
                const MissingCacheElement& element = getMissingCacheElement_(missingTrain_[i]);

                for (uint_t k = 0, l = 0; k < d; ++k) {
                    if (!element.missing[k]) {
                        int o = element.indexObserved[k];

                        double delta = inputTrain_(i,k) - parameters_.basisFunctionPositions(j,k); // GPzMatLab: Delta(i,:)
                        double deltaSolved; // GPzMatLab: delta/Sigma(o,o) or delta/iPSoo
                        double derivInvCovarianceScalar; // GPzMatLab: diSoo

                        if (inputErrorTrain_.rows() == 0) {
                            deltaSolved = element.invCovariancesObserved[j](o,o)*delta;
                            derivInvCovarianceScalar = (-0.5*derivBasis(i,j))*square(delta);
                        } else {
                            double covarianceScalar = element.covariancesObserved[j](o,o) + inputErrorTrain_(i,k);
                            deltaSolved = delta/covarianceScalar;
                            derivInvCovarianceScalar = (-0.5*derivBasis(i,j))*element.covariancesObserved[j](o,o)*(
                                1 + element.covariancesObserved[j](o,o)*(square(deltaSolved) - 1.0/covarianceScalar)
                            );
                        }

                        // Derivative wrt to basis positions
                        // =================================
                        derivatives_.basisFunctionPositions(j,k) += derivBasis(i,j)*deltaSolved;

                        // Derivative wrt to basis covariances
                        // =================================
                        double dgO = element.dgO[j](k,l)*derivInvCovarianceScalar;
                        derivatives_.basisFunctionCovariances[j](k,k) += dgO;

                        // NB: no contribution from gUO because it is just zero (no covariance)

                        ++l;
                    }
                }
            }
        };

        if ((d == 1 && optimizations_.specializeForSingleFeature) ||
            (diagonalCovariance && optimizations_.specializeForDiagCovariance)) {
            pool.execute(derivativeAddContributionDiag, m);
        } else {
            pool.execute(derivativeAddContribution, m);
        }

        if (profileTraining_) {
            timeNow = now();
            std::cout << "updateDerivatives 3 " << timeNow - timePrev << std::endl;
            std::cout << "derivative: " << derivatives_.basisFunctionCovariances[0](0,0) << std::endl;
            timePrev = timeNow;
        }
    }
}

void GPz::updateLikelihoodValid_() {
    const uint_t n = inputValid_.rows();

    // Pre-compute things
    updateValidBasisFunctions_();
    updateValidOutputErrors_();

    // Do the hard work... (simple Gaussian likelihood (model-obs)/errors ~ chi2)
    Mat1d deviates = validBasisFunctions_*modelWeights_ - outputValid_.matrix(); // GPzMatLab: delta

    logLikelihoodValid_ = 0.0;
    for (uint_t i = 0; i < n; ++i) {
        logLikelihoodValid_ -= weightValid_[i]*0.5*
            (exp(-validOutputLogError_[i])*deviates[i]*deviates[i] + validOutputLogError_[i]);
    }

    logLikelihoodValid_ /= n;
    logLikelihoodValid_ -= 0.5*log(2.0*M_PI);
}

void GPz::computeInputPriors_() {
    const uint_t m = numberBasisFunctions_;
    const uint_t n = inputTrain_.rows();
    const uint_t d = numberFeatures_;

    modelInputPrior_.resize(m);
    modelInputPrior_.fill(1.0/m);

    // Build the components of the gaussian mixture model that is used to
    // fit the probability distribution of the input data.
    // The GMM components Gaussians are taken to be the same as the best fit basis functions.
    Mat2d gaussianMixtureComponents(n,m);
    for (uint_t i = 0; i < n; ++i) {
        const MissingCacheElement& element = getMissingCacheElement_(missingTrain_[i]);

        for (uint_t j = 0; j < m; ++j) {
            // Reuse the basis functions, albeit normalized properly this time.
            // (the GP doesn't care about the basis function normalization, but the GMM does).
            double value = element.covariancesObservedLogDeterminant[j];
            value += (d - element.countMissing)*log(2.0*M_PI);
            value -= element.countMissing*log(2.0);

            gaussianMixtureComponents(i,j) = exp(log(trainBasisFunctions_(i,j)) - 0.5*value);
        }
    }

    // Iterative procedure to compute the priors on Gaussian Mixture model weights.
    // This is solved using an Expectation-Maximization algorithm.
    const uint_t maxPriorIterations = 100;
    for (uint_t iter = 0; iter < maxPriorIterations; ++iter) {
        Mat1d oldPrior = modelInputPrior_;

        Mat2d weight = gaussianMixtureComponents;
        for (uint_t i = 0; i < n; ++i) {
            weight.array().row(i) *= modelInputPrior_.array().transpose();
            double sum = weight.row(i).sum();
            weight.row(i) /= sum;
        }

        modelInputPrior_ = weight.colwise().mean();

        double delta = (modelInputPrior_ - oldPrior).norm()/(modelInputPrior_ + oldPrior).norm();

        if (delta < 1e-10) {
            // Converged
            break;
        }
    }

    if (modelInputPrior_.sum() <= 0.0) {
        throw std::runtime_error("prior is zero");
    }
}

// ==============================
// Internal functions: prediction
// ==============================

void GPz::predictFull_(const Mat1d& input, const MissingCacheElement& element, double& value,
    double& varianceTrainDensity, double& varianceTrainNoise) const {

    // The simplest GPz case: no missing data, no noisy inputs

    Mat1d basis = evaluateBasisFunctions_(input, Mat1d{}, element);

    value = basis.transpose()*modelWeights_; // GPzMatLab: mu
    varianceTrainDensity = basis.transpose()*modelInvCovariance_*basis; // GPzMatLab: nu
    varianceTrainNoise = exp(evaluateOutputLogError_(basis)); // GPzMatLab: beta_i
}

void GPz::predictNoisy_(const Mat1d& input, const Mat1d& inputError,
    const MissingCacheElement& element, double& value,
    double& varianceTrainDensity, double& varianceTrainNoise, double& varianceInputNoise) const {

    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    const bool diagonalCovariance = covarianceType_ == CovarianceType::VARIABLE_DIAGONAL ||
                                    covarianceType_ == CovarianceType::GLOBAL_DIAGONAL;

    // No missing data, but we have noisy inputs

    Mat1d basis = evaluateBasisFunctions_(input, inputError, element);

    value = basis.transpose()*modelWeights_; // GPzMatLab: mu

    varianceTrainDensity = 0.0; // GPzMatLab: nu
    varianceInputNoise = 0.0; // GPzMatLab: gamma
    double VlnS = 0.0;

    if ((d == 1 && optimizations_.specializeForSingleFeature) ||
        (diagonalCovariance && optimizations_.specializeForDiagCovariance)) {
        // Specialized version for one feature or diagonal covariances (faster)

        for (uint_t i = 0; i < m; ++i)
        for (uint_t j = 0; j <= i; ++j) {
            double lnNxc = 0.0;
            double det = 1.0;
            for (uint_t k = 0; k < d; ++k) {
                double icovi = element.invCovariancesObserved[i](k,k);
                double icovj = element.invCovariancesObserved[j](k,k);
                double posi = parameters_.basisFunctionPositions(i,k);
                double posj = parameters_.basisFunctionPositions(j,k);

                double Cij = 1.0/(icovi + icovj);
                double cij = Cij*(posi*icovi + posj*icovj);

                double Delta = input[k] - cij;

                Cij += inputError[k];
                lnNxc += square(Delta)/Cij;
                det *= Cij;
            }

            double ZijNxc = exp(lnZ(i,j) - 0.5*lnNxc)/sqrt(det);

            double coef = (i == j ? 1.0 : 2.0)*ZijNxc;
            varianceTrainDensity += coef*modelInvCovariance_(i,j);
            varianceInputNoise += coef*modelWeights_[i]*modelWeights_[j];
            VlnS += coef*parameters_.uncertaintyBasisWeights[i]*parameters_.uncertaintyBasisWeights[j];
        }
    } else {
        // Generic version for any number of features and any covariance type
        Eigen::JacobiSVD<Mat2d> svd;
        Mat2d iCij;
        Mat2d Cij;
        Mat1d cij;
        Mat1d Delta;
        Mat1d DeltaSolved;

        for (uint_t i = 0; i < m; ++i)
        for (uint_t j = 0; j <= i; ++j) {
            iCij = element.invCovariancesObserved[i] + element.invCovariancesObserved[j];

            svd.compute(iCij, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Cij = computeInverseSymmetric(iCij, svd);
            cij = parameters_.basisFunctionPositions.row(i)*element.invCovariancesObserved[i]
                + parameters_.basisFunctionPositions.row(j)*element.invCovariancesObserved[j];

            cij = svd.solve(cij);

            // Now this is source-specific

            Delta = input - cij;
            Cij.diagonal() += inputError;
            svd.compute(Cij, Eigen::ComputeThinU | Eigen::ComputeThinV);

            DeltaSolved = svd.solve(Delta);
            double lnNxc = DeltaSolved.transpose()*Delta + computeLogDeterminant(svd);
            double ZijNxc = exp(lnZ(i,j) - 0.5*lnNxc);

            double coef = (i == j ? 1.0 : 2.0)*ZijNxc;
            varianceTrainDensity += coef*modelInvCovariance_(i,j);
            varianceInputNoise += coef*modelWeights_[i]*modelWeights_[j];
            VlnS += coef*parameters_.uncertaintyBasisWeights[i]*parameters_.uncertaintyBasisWeights[j];
        }
    }

    varianceInputNoise -= value*value;

    double logError = evaluateOutputLogError_(basis);
    VlnS -= square(logError - parameters_.logUncertaintyConstant);
    varianceTrainNoise = exp(logError)*(1.0 + 0.5*VlnS); // GPzMatLab: beta_i
}

void GPz::predictMissingNoisy_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element, double& value,
    double& varianceTrainDensity, double& varianceTrainNoise, double& varianceInputNoise) const {

    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;
    const bool noError = inputError.rows() == 0;
    const bool diagonalCovariance = covarianceType_ == CovarianceType::VARIABLE_DIAGONAL ||
                                    covarianceType_ == CovarianceType::GLOBAL_DIAGONAL;

    Mat1d Pio(m); // GPzMatLab: Ex & Pio (combined)
    Mat1d No(m); // GPzMatLab: No

    Mat2d filledInput; // GPzMatLab: X_hat
    std::vector<Mat2d> Psi_hat; // GPzMatLab: Psi_hat
    if (!diagonalCovariance || !optimizations_.specializeForDiagCovariance) {
        filledInput.resize(d,m);
        Psi_hat.resize(m);
    }

    for (uint_t i = 0; i < m; ++i) {
        if (diagonalCovariance && optimizations_.specializeForDiagCovariance) {
            if (element.countMissing < d) {
                double lnNo = 0.0;
                double det = 1.0;
                for (uint_t k = 0; k < d; ++k) {
                    if (!element.missing[k]) {
                        int o = element.indexObserved[k];
                        double Delta = input[k] - parameters_.basisFunctionPositions(i,k);
                        double sigma = element.covariancesObserved[i](o,o);
                        if (!noError) {
                            sigma += inputError[k];
                        }

                        lnNo += square(Delta)/sigma;
                        det *= sigma;
                    }
                }

                No[i] = exp(-0.5*lnNo)/sqrt(det);
                Pio[i] = No[i]*modelInputPrior_[i];
            } else {
                No[i] = 1.0;
                Pio[i] = modelInputPrior_[i];
            }
        } else {
            Mat1d position = parameters_.basisFunctionPositions.row(i);
            Mat1d Delta = input - position;
            Mat1d DeltaObserved;
            fetchVectorElements_(DeltaObserved, Delta, element, 'o');

            if (element.countMissing < d) {
                Mat2d sigma = element.covariancesObserved[i];

                if (!noError) {
                    Mat1d varianceObserved;
                    fetchVectorElements_(varianceObserved, inputError, element, 'o');
                    sigma.diagonal() += varianceObserved;
                    Psi_hat[i] = element.T[i]*varianceObserved.asDiagonal()*element.T[i].transpose();
                } else {
                    Psi_hat[i].setZero(d,d);
                }

                addMatrixElements_(Psi_hat[i], element.Psi_hat[i], element, 'u', 'u');

                Eigen::JacobiSVD<Mat2d> svd(sigma, Eigen::ComputeThinU | Eigen::ComputeThinV);
                double lnNo = svd.solve(DeltaObserved).transpose()*DeltaObserved
                            + computeLogDeterminant(svd);
                No[i] = exp(-0.5*lnNo);
                Pio[i] = No[i]*modelInputPrior_[i];
            } else {
                Psi_hat[i].setZero(d,d);
                addMatrixElements_(Psi_hat[i], element.Psi_hat[i], element, 'u', 'u');

                No[i] = 1.0;
                Pio[i] = modelInputPrior_[i];
            }

            Mat1d positionMissing; // GPzMatLab: P(i,~o)
            fetchVectorElements_(positionMissing, position, element, 'u');

            Mat1d inputMissing = positionMissing + element.R[i].transpose()*DeltaObserved;
            for (uint_t j = 0; j < d; ++j) {
                if (element.missing[j]) {
                    int u = element.indexMissing[j];
                    filledInput(j,i) = inputMissing[u];
                } else {
                    filledInput(j,i) = input[j];
                }
            }
        }
    }

    Pio /= Pio.sum();

    Mat1d basis; // GPzMatLab: PHI
    basis.setZero(m);

    varianceTrainDensity = 0.0; // GPzMatLab: nu
    varianceInputNoise = 0.0; // GPzMatLab: gamma
    double VlnS = 0.0;

    if (diagonalCovariance && optimizations_.specializeForDiagCovariance) {
        for (uint_t i = 0; i < m; ++i) {
            double value = 0.0;
            for (uint_t j = 0; j < m; ++j) {
                value += element.Nij(i,j)*Pio[j];
            }

            basis[i] = No[i]*value;
        }

        for (uint_t i = 0; i < m; ++i)
        for (uint_t j = 0; j <= i; ++j) {
            double EcCij = 0.0;
            for (uint_t l = 0; l < m; ++l) {
                EcCij += element.Nu[i][j][l]*Pio[l];
            }

            double det = 1.0;
            double lnN = 0.0;
            for (uint_t k = 0; k < d; ++k) {
                if (!element.missing[k]) {
                    double icovi = noMissingCache_->invCovariancesObserved[i](k,k);
                    double icovj = noMissingCache_->invCovariancesObserved[j](k,k);
                    double posi = parameters_.basisFunctionPositions(i,k);
                    double posj = parameters_.basisFunctionPositions(j,k);

                    double Cij = 1.0/(icovi + icovj);
                    double cij = (posi*icovi + posj*icovj)*Cij;

                    if (!noError) {
                        Cij += inputError[k];
                    }

                    double Delta = input[k] - cij;
                    lnN += square(Delta)/Cij;
                    det *= Cij;
                }
            }

            EcCij *= exp(-0.5*lnN)/sqrt(det);

            double Pij = exp(lnZ(i,j))*EcCij;

            double coef = (i == j ? 1.0 : 2.0)*Pij;
            varianceTrainDensity += coef*modelInvCovariance_(i,j);
            varianceInputNoise += coef*modelWeights_[i]*modelWeights_[j];
            VlnS += coef*parameters_.uncertaintyBasisWeights[i]*parameters_.uncertaintyBasisWeights[j];
        }
    } else {
        Eigen::JacobiSVD<Mat2d> svd;
        Mat2d iCij;
        Mat2d Cij;
        Mat1d cij;
        Mat1d Delta;
        Mat1d DeltaSolved;
        Mat2d Sij;

        for (uint_t i = 0; i < m; ++i)
        for (uint_t j = 0; j <= i; ++j) {
            iCij = noMissingCache_->invCovariancesObserved[i] + noMissingCache_->invCovariancesObserved[j];

            svd.compute(iCij, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Cij = computeInverseSymmetric(iCij, svd);
            cij = parameters_.basisFunctionPositions.row(i)*noMissingCache_->invCovariancesObserved[i]
                + parameters_.basisFunctionPositions.row(j)*noMissingCache_->invCovariancesObserved[j];

            cij = svd.solve(cij);

            Delta = filledInput.col(j) - parameters_.basisFunctionPositions.row(i).transpose();
            Sij = noMissingCache_->covariancesObserved[i] + Psi_hat[j];
            svd.compute(Sij, Eigen::ComputeThinU | Eigen::ComputeThinV);

            DeltaSolved = svd.solve(Delta);
            double N = exp(-0.5*DeltaSolved.transpose()*Delta - 0.5*computeLogDeterminant(svd));
            basis[i] += N*Pio[j];

            if (i != j) {
                Delta = filledInput.col(i) - parameters_.basisFunctionPositions.row(j).transpose();
                Sij = noMissingCache_->covariancesObserved[j] + Psi_hat[i];
                svd.compute(Sij, Eigen::ComputeThinU | Eigen::ComputeThinV);

                DeltaSolved = svd.solve(Delta);
                N = exp(-0.5*DeltaSolved.transpose()*Delta - 0.5*computeLogDeterminant(svd));
                basis[j] += N*Pio[i];
            }

            double EcCij = 0.0;
            for (uint_t l = 0; l < m; ++l) {
                Delta = filledInput.col(l) - cij;
                Sij = Cij + Psi_hat[l];
                svd.compute(Sij, Eigen::ComputeThinU | Eigen::ComputeThinV);

                DeltaSolved = svd.solve(Delta);
                N = exp(-0.5*DeltaSolved.transpose()*Delta - 0.5*computeLogDeterminant(svd));

                EcCij += N*Pio[l];
            }

            double Pij = exp(lnZ(i,j))*EcCij;

            double coef = (i == j ? 1.0 : 2.0)*Pij;
            varianceTrainDensity += coef*modelInvCovariance_(i,j);
            varianceInputNoise += coef*modelWeights_[i]*modelWeights_[j];
            VlnS += coef*parameters_.uncertaintyBasisWeights[i]*parameters_.uncertaintyBasisWeights[j];
        }
    }

    basis.array() *= exp(0.5*noMissingCache_->covariancesObservedLogDeterminant);

    value = basis.transpose()*modelWeights_; // GPzMatLab: mu

    varianceInputNoise -= value*value;

    double logError = evaluateOutputLogError_(basis);
    VlnS -= square(logError - parameters_.logUncertaintyConstant);
    varianceTrainNoise = exp(logError)*(1.0 + 0.5*VlnS); // GPzMatLab: beta_i
}

GPzOutput GPz::predict_(const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const {
    const uint_t n = input.rows();
    const uint_t m = numberBasisFunctions_;
    const bool noError = inputError.rows() == 0;

    GPzOutput result;
    result.value.setZero(n);
    result.varianceTrainDensity.setZero(n);
    result.varianceTrainNoise.setZero(n);
    result.varianceInputNoise.setZero(n);

    // Compute cached variables
    lnZ.setZero(m,m); {
        for (uint_t i = 0; i < m; ++i)
        for (uint_t j = 0; j <= i; ++j) {
            Mat2d Sij = noMissingCache_->covariancesObserved[i]
                +noMissingCache_->covariancesObserved[j];

            Eigen::JacobiSVD<Mat2d> svd(Sij, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Mat1d Delta = parameters_.basisFunctionPositions.row(i)
                        - parameters_.basisFunctionPositions.row(j);

            lnZ(i,j) = +0.5*noMissingCache_->covariancesObservedLogDeterminant[i]
                +0.5*noMissingCache_->covariancesObservedLogDeterminant[j]
                -0.5*svd.solve(Delta).transpose()*Delta
                -0.5*computeLogDeterminant(svd);
        }
    }

    // Iterate over galaxies
    auto doPrediction = [&](uint_t i) {
        const MissingCacheElement& element = getMissingCacheElement_(missing[i]);

        if (element.countMissing == 0) {
            if (noError) {
                predictFull_(input.row(i), element, result.value[i], result.varianceTrainDensity[i],
                    result.varianceTrainNoise[i]);

                result.varianceInputNoise[i] = 0.0;
            } else {
                predictNoisy_(input.row(i), inputError.row(i), element, result.value[i],
                    result.varianceTrainDensity[i], result.varianceTrainNoise[i],
                    result.varianceInputNoise[i]);
            }
        } else {
            if (noError) {
                predictMissingNoisy_(input.row(i), Mat1d{}, element, result.value[i],
                    result.varianceTrainDensity[i], result.varianceTrainNoise[i],
                    result.varianceInputNoise[i]);

                // NB: varianceInputNoise includes noise contribution from missing variables
            } else {
                predictMissingNoisy_(input.row(i), inputError.row(i), element, result.value[i],
                    result.varianceTrainDensity[i], result.varianceTrainNoise[i],
                    result.varianceInputNoise[i]);
            }
        }
    };

    parallel_for pool(optimizations_.enableMultithreading ? optimizations_.maxThreads : 0);

    if (verbose_) {
        double timeStart = now();
        pool.callback = [&,timeStart](uint_t iter) {
            double timeSpent = now() - timeStart;
            double timeLeft = timeSpent/float(iter)*(n - iter);

            uint_t defaultWidth = std::cout.width();
            std::cout << "predicted " << std::setw(log10(n)+2) << iter << std::setw(defaultWidth);
            std::cout << "/" << n << " (" << floor(iter/float(n)*100) << "%), "
                << "time remaining: ";

            if (!std::isfinite(timeLeft)) {
                std::cout << "unknown";
            } else {
                uint_t numHours = floor(timeLeft/3600.0);
                if (numHours > 0) {
                    std::cout << numHours << "h";
                    timeLeft -= numHours*3600.0;
                }
                uint_t numMinutes = floor(timeLeft/60.0);
                if (numMinutes > 0) {
                    std::cout << numMinutes << "m";
                    timeLeft -= numMinutes*60.0;
                }
                std::cout << floor(timeLeft*100)/100 << "s";
            }

            std::cout << std::endl;
        };
    }

    pool.execute(doPrediction, n);

    result.variance = result.varianceTrainDensity + result.varianceTrainNoise
        + result.varianceInputNoise;

    return result;
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
        if (newFunction == PriorMeanFunction::LINEAR_MARGINALIZE ||
            newFunction != PriorMeanFunction::LINEAR_PREPROCESS) {
            throw std::runtime_error("not implemented");
        }

        priorMean_ = newFunction;
        reset_();
    }
}

PriorMeanFunction GPz::getPriorMeanFunction() const {
    return priorMean_;
}

void GPz::setNumberOfFeatures_(uint_t num) {
    if (num != numberFeatures_) {
        if (verbose_) {
            std::cout << "setting number of features to " << num << std::endl;
        }

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

void GPz::setBalancedWeightingBinSize(double size) {
    balancedWeightingBinSize_ = size;
}

double GPz::getBalancedWeightingBinSize() const {
    return balancedWeightingBinSize_;
}

void GPz::setNormalizationScheme(NormalizationScheme scheme) {
    normalizationScheme_ = scheme;
}

NormalizationScheme GPz::getNormalizationScheme() const {
    return normalizationScheme_;
}

void GPz::setTrainValidationSplitMethod(TrainValidationSplitMethod method) {
    trainValidSplitMethod_ = method;
}

TrainValidationSplitMethod GPz::getTrainValidationSplitMethod() const {
    return trainValidSplitMethod_;
}

void GPz::setTrainValidationRatio(double ratio) {
    trainValidRatio_ = ratio;
}

double GPz::getTrainValidationRatio() const {
    return trainValidRatio_;
}

void GPz::setTrainValidationSplitSeed(uint_t seed) {
    seedTrainSplit_ = seed;
}

uint_t GPz::getTrainValidationSplitSeed() const {
    return seedTrainSplit_;
}

void GPz::setInitialPositionSeed(uint_t seed) {
    seedPositions_ = seed;
}

uint_t GPz::getInitialPositionSeed() const {
    return seedPositions_;
}

void GPz::setCovarianceType(CovarianceType type) {
    covarianceType_ = type;
}

CovarianceType GPz::getCovarianceType() const {
    return covarianceType_;
}

void GPz::setOutputUncertaintyType(OutputUncertaintyType type) {
    outputUncertaintyType_ = type;
}

OutputUncertaintyType GPz::getOutputUncertaintyType() const {
    return outputUncertaintyType_;
}

void GPz::setOptimizerMethod(OptimizerMethod method) {
    optimizerMethod_ = method;
}

OptimizerMethod GPz::getOptimizerMethod() const {
    return optimizerMethod_;
}

void GPz::setOptimizationMaxIterations(uint_t maxIterations) {
    optimizationMaxIterations_ = maxIterations;
}

uint_t GPz::getOptimizationMaxIterations() const {
    return optimizationMaxIterations_;
}

void GPz::setOptimizationTolerance(double tolerance) {
    optimizationTolerance_ = tolerance;
}

double GPz::getOptimizationTolerance() const {
    return optimizationTolerance_;
}

void GPz::setOptimizationGradientTolerance(double tolerance) {
    optimizationGradientTolerance_ = tolerance;
}

double GPz::getOptimizationGradientTolerance() const {
    return optimizationGradientTolerance_;
}

void GPz::setVerboseMode(bool verbose) {
    verbose_ = verbose;
}

bool GPz::getVerboseMode() const {
    return verbose_;
}

void GPz::setOptimizationFlags(GPzOptimizations optimizations) {
    optimizations_ = optimizations;
}

GPzOptimizations GPz::getOptimizationFlags() const {
    return optimizations_;
}

void GPz::setProfileTraining(bool profile) {
    profileTraining_ = profile;
}

// =====================
// Fit/training function
// =====================

void GPz::fit(Mat2d input, Mat2d inputError, Vec1d output) {
    // Check inputs are consistent
    if (!checkErrorDimensions_(input, inputError)) {
        throw std::runtime_error("input uncertainty has incorrect dimension");
    }

    auto start = std::chrono::steady_clock::now();

    if (verbose_) {
        std::cout << "begin fitting with " << input.rows() << " data points" << std::endl;
        std::cout << "found " << input.cols() << " features" << std::endl;
        if (inputError.rows() != 0) {
            std::cout << "found uncertainties for the features" << std::endl;
        }
    }

    // Normalize the inputs
    setNumberOfFeatures_(input.cols());
    initializeInputs_(std::move(input), std::move(inputError), std::move(output));

    // Setup the fit, initialize arrays, etc.
    initializeFit_();

    // Build vector with initial values for hyper-parameter
    Vec1d initialValues = makeParameterArray_(parameters_);

    if (verbose_) {
        std::cout << "starting optimization of model" << std::endl;
    }

    // Minimization function
    auto minFunc = [this](const Vec1d& vectorParameters, Minimize::FunctionOutput requested) {
        if (requested == Minimize::FunctionOutput::METRIC_VALID) {
            // Compute likelihood of the validation set
            updateLikelihoodValid_();

            // Return only the log likelihood
            Vec1d result(1);
            result[0] = -logLikelihoodValid_;

            return result;
        } else {
            // Load new parameters
            loadParametersArray_(vectorParameters, parameters_);

            // Update model
            updateTrainModel_(requested);

            Vec1d result(1+numberParameters_);

            if (requested == Minimize::FunctionOutput::ALL_TRAIN ||
                requested == Minimize::FunctionOutput::METRIC_TRAIN) {
                // Return log likelihood
                result[0] = -logLikelihood_/inputTrain_.rows();
            }

            if (requested == Minimize::FunctionOutput::ALL_TRAIN ||
                requested == Minimize::FunctionOutput::DERIVATIVES_TRAIN) {
                // Return derivatives
                Vec1d vectorDerivatives = makeParameterArray_(derivatives_);
                for (uint_t i = 0; i < numberParameters_; ++i) {
                    result[1+i] = -vectorDerivatives[i]/inputTrain_.rows();
                }
            }

            return result;
        }
    };

    // Use BFGS for minimization
    Minimize::Options options;
    options.maxIterations = optimizationMaxIterations_;
    options.hasValidation = inputValid_.rows() != 0;
    options.minimizerTolerance = optimizationTolerance_;
    options.gradientTolerance = optimizationGradientTolerance_;
    options.verbose = verbose_;

    Minimize::Result result;
    if (optimizerMethod_ == OptimizerMethod::GPZ_LBFGS) {
        result = Minimize::minimizeLBFGS(options, initialValues, minFunc);
    } else {
        #ifdef NO_GSL
        throw std::runtime_error("GSL minimizer is not available");
        #else
        result = Minimize::minimizeBFGS(options, initialValues, minFunc);
        #endif
    }

    if (!result.success) {
        throw std::runtime_error("minimization failed");
    }

    if (inputValid_.rows() == 0) {
        // No validation set, use the latest best model
        loadParametersArray_(result.parameters, parameters_);
    } else {
        // Validation set provided, use the latest best model from the validation
        loadParametersArray_(result.parametersBestValid, parameters_);
    }

    // Update model to best parameter set (and likelihood, for diagnostic)
    updateTrainModel_(Minimize::FunctionOutput::METRIC_TRAIN);

    // Update likelihood of the validation set (for diagnostic)
    updateLikelihoodValid_();

    // Compute priors of input data distribution for predictions with missing variables
    computeInputPriors_();

    auto end = std::chrono::steady_clock::now();

    if (verbose_) {
        std::cout << "total time required for fit: " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1e3
            << " seconds" << std::endl;
    }
}

// ========================
// Fit model loading/saving
// ========================

void GPz::loadModel(const GPzModel& model) {
    int tmpNumBF = model.modelWeights.size();
    int tmpNumFeature = model.featureMean.size();

    if (model.featureMean.size() != tmpNumFeature) {
        throw std::runtime_error("wrong size for featureMean");
    }
    if (model.featureSigma.size() != tmpNumFeature) {
        throw std::runtime_error("wrong size for featureSigma");
    }
    if (model.modelWeights.size() != tmpNumBF) {
        throw std::runtime_error("wrong size for modelWeights");
    }
    if (model.modelInputPrior.size() != tmpNumBF) {
        throw std::runtime_error("wrong size for modelInputPrior");
    }
    if (model.modelInvCovariance.rows() != tmpNumBF) {
        throw std::runtime_error("wrong number of rows for modelInvCovariance");
    }
    if (model.modelInvCovariance.cols() != tmpNumBF) {
        throw std::runtime_error("wrong number of columns for modelInvCovariance");
    }
    if (model.parameters.basisFunctionPositions.rows() != tmpNumBF) {
        throw std::runtime_error("wrong number of rows for basisFunctionPositions");
    }
    if (model.parameters.basisFunctionPositions.cols() != tmpNumFeature) {
        throw std::runtime_error("wrong number of columns for basisFunctionPositions");
    }
    if (model.parameters.basisFunctionLogRelevances.size() != tmpNumBF) {
        throw std::runtime_error("wrong number of columns for basisFunctionLogRelevances");
    }
    if (model.parameters.basisFunctionCovariances.size() != static_cast<uint_t>(tmpNumBF)) {
        throw std::runtime_error("wrong size for basisFunctionCovariances");
    }
    if (model.parameters.basisFunctionCovariances[0].rows() != tmpNumFeature) {
        throw std::runtime_error("wrong number of rows for basisFunctionCovariances");
    }
    if (model.parameters.basisFunctionCovariances[0].cols() != tmpNumFeature) {
        throw std::runtime_error("wrong number of columns for basisFunctionCovariances");
    }
    if (model.parameters.uncertaintyBasisWeights.size() != tmpNumBF) {
        throw std::runtime_error("wrong size for uncertaintyBasisWeights");
    }
    if (model.parameters.uncertaintyBasisWeights.size() != tmpNumBF) {
        throw std::runtime_error("wrong size for uncertaintyBasisLogRelevances");
    }

    double epsilon = 1e-9;

    for (int i = 0; i < tmpNumFeature; ++i) {
        if (!std::isfinite(model.featureMean[i])) {
            throw std::runtime_error("feature mean has invalid value (not finite)");
        }
        if (!std::isfinite(model.featureSigma[i]) || model.featureSigma[i] < 0.0) {
            throw std::runtime_error("feature sigma has invalid value (not finite or negative)");
        }
    }

    if (!std::isfinite(model.outputMean)) {
        throw std::runtime_error("output mean has invalid value (not finite)");
    }

    for (int i = 0; i < tmpNumBF; ++i) {
        for (int j = 0; j < tmpNumBF; ++j) {
            if (!std::isfinite(model.modelInvCovariance(i,j))) {
                throw std::runtime_error("model inverse covariance has invalid value (not finite)");
            }
            if (std::abs(model.modelInvCovariance(i,j) - model.modelInvCovariance(j,i)) > epsilon) {
                throw std::runtime_error("model inverse covariance is not symmetric");
            }
        }
        if (!std::isfinite(model.modelInputPrior[i]) || model.modelInputPrior[i] < 0.0) {
            throw std::runtime_error("model prior has invalid value (not finite or negative)");
        }
        for (int j = 0; j < tmpNumFeature; ++j) {
            if (!std::isfinite(model.parameters.basisFunctionPositions(i,j))) {
                throw std::runtime_error("basis function position has invalid value (not finite)");
            }
        }
        if (!std::isfinite(model.parameters.basisFunctionLogRelevances[i])) {
            throw std::runtime_error("BF log-relevance has invalid value (not finite)");
        }
        for (int j = 0; j < tmpNumFeature; ++j)
        for (int k = 0; k < tmpNumFeature; ++k) {
            auto& cov = model.parameters.basisFunctionCovariances[i];
            if (!std::isfinite(cov(j,k))) {
                throw std::runtime_error("BF covariance has invalid value (not finite)");
            }
            if (std::abs(cov(j,k) - cov(k,j)) > epsilon) {
                throw std::runtime_error("BF covariance is not symmetric");
            }
        }
        if (!std::isfinite(model.parameters.uncertaintyBasisWeights[i])) {
            throw std::runtime_error("BF uncertainty weight has invalid value (not finite)");
        }
        if (!std::isfinite(model.parameters.uncertaintyBasisLogRelevances[i])) {
            throw std::runtime_error("BF uncertainty log-relevance has invalid value (not finite)");
        }
    }

    if (!std::isfinite(model.parameters.logUncertaintyConstant)) {
        throw std::runtime_error("log uncertainty has invalid value (not finite)");
    }

    setNumberOfBasisFunctions(tmpNumBF);
    setNumberOfFeatures_(tmpNumFeature);

    setPriorMeanFunction(model.priorMean);
    setOutputUncertaintyType(model.outputUncertaintyType);
    setNormalizationScheme(model.normalizationScheme);

    featureMean_ = model.featureMean;
    featureSigma_ = model.featureSigma;
    outputMean_ = model.outputMean;

    parameters_ = model.parameters;
    modelWeights_ = model.modelWeights;
    modelInvCovariance_ = model.modelInvCovariance;
    modelInputPrior_ = model.modelInputPrior;
}

GPzModel GPz::getModel() const {
    GPzModel model;

    model.priorMean = priorMean_;
    model.outputUncertaintyType = outputUncertaintyType_;
    model.normalizationScheme = normalizationScheme_;

    model.featureMean = featureMean_;
    model.featureSigma = featureSigma_;
    model.outputMean = outputMean_;

    model.parameters = parameters_;
    model.modelWeights = modelWeights_;
    model.modelInvCovariance = modelInvCovariance_;
    model.modelInputPrior = modelInputPrior_;

    return model;
}

// ===================
// Prediction function
// ===================

GPzOutput GPz::predict(Mat2d input, Mat2d inputError) const {
    // Check input is consistent
    if (!checkInputDimensions_(input)) {
        throw std::runtime_error("input has incorrect dimension");
    }
    if (!checkErrorDimensions_(input, inputError)) {
        throw std::runtime_error("input uncertainty has incorrect dimension");
    }

    // Check that we have a usable set of parameters to make predictions
    if (parameters_.basisFunctionPositions.rows() == 0) {
        throw std::runtime_error("model is not initialized");
    }

    auto start = std::chrono::steady_clock::now();

    if (verbose_) {
        std::cout << "begin prediction for " << input.rows() << " data points" << std::endl;
        std::cout << "found " << input.cols() << " features" << std::endl;
        if (inputError.rows() != 0) {
            std::cout << "found uncertainties for the features" << std::endl;
        }
        std::cout << "precomputing cache and applying normalization" << std::endl;
    }

    // Detect missing data
    buildMissingCache_(input);
    updateMissingCache_(MissingCacheUpdate::PREDICT);

    Vec1i missing = getBestMissingID_(input);

    // Project input from real space to training space
    applyInputNormalization_(input, inputError);

    if (verbose_) {
        std::cout << "starting prediction" << std::endl;
    }

    // Make prediction
    GPzOutput result = predict_(input, inputError, missing);

    if (verbose_) {
        std::cout << "renormalize output values" << std::endl;
    }

    // De-project output from training space to real space
    restoreOutputNormalization_(input, missing, result);

    auto end = std::chrono::steady_clock::now();

    if (verbose_) {
        std::cout << "total time required for prediction: " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1e3
            << " seconds" << std::endl;
    }

    return result;
}

}  // namespace PHZ_GPz


