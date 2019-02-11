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
#include <random>
#include <iostream>
#include <Eigen/Eigenvalues>

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

void GPz::resizeHyperParameters_(HyperParameters& params) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    params.basisFunctionPositions.resize(m,d);
    params.basisFunctionLogRelevances.resize(m);
    params.basisFunctionCovariances.resize(m);
    for (uint_t i = 0; i < m; ++i) {
        params.basisFunctionCovariances[i].resize(d,d);
    }

    // TODO: Save memory: basisFunctionCovariances only needs one element
    // when the covariance type is any of the GLOBAL_* types.

    params.uncertaintyBasisWeights.resize(m);
    params.uncertaintyBasisLogRelevances.resize(m);
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
        featureMean_.resize(d);
        featureSigma_.resize(d);
    }

    featurePCAMean_.resize(d);
    featurePCASigma_.resize(d,d);
    featurePCABasisVectors_.resize(d,d);

    if (verbose_) {
        std::cout << "allocated memory for fit parameters" << std::endl;
    }
}

void GPz::reset_() {
    parameters_ = HyperParameters{};
    derivatives_ = HyperParameters{};

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
        assert(numberTrain != 0 && "cannot have zero training data points");

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
            weight = pow(1.0/(1.0 + output), 2);
            break;
        }
        case WeightingScheme::BALANCED: {
            // Make bins
            double minValue = output.minCoeff();
            double maxValue = output.maxCoeff();
            uint_t numBins = ceil((maxValue - minValue)/balancedWeightingBinSize_);

            Vec1d bins(numBins+1);
            std::iota(std::begin(bins), std::end(bins), 0.0);
            bins = bins*balancedWeightingBinSize_ + minValue;

            // Compute histogram of counts in bins
            uint_t maxCount = 0;
            weight.resize(output.rows());
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

    return weight;
}

void GPz::initializeInputs_(Mat2d input, Mat2d inputError, Vec1d output) {
    // Compute weights and split training/valid
    Vec1d weight = computeWeights_(output);
    splitTrainValid_(input, inputError, output, weight);

    // Setup missing cache
    buildMissingCache_(inputTrain_);
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
    assert(solver.info() == Eigen::Success && "could not get eigenvectors of PCA sigma matrix");

    Mat1d eigenValues = solver.eigenvalues().real();
    Mat2d eigenVectors = solver.eigenvectors().real();

    eigenValues = (eigenValues/(n-1.0)).cwiseSqrt();
    featurePCABasisVectors_ = eigenValues.asDiagonal()*eigenVectors.transpose();

    featurePCASigma_ /= n;
}

void GPz::initializeBasisFunctions_() {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

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
    }
}

void GPz::initializeBasisFunctionRelevances_() {
    const uint_t n = outputTrain_.rows();
    double outputLogVariance = log(outputTrain_.square().sum()/(n-1.0));
    parameters_.basisFunctionLogRelevances.fill(-outputLogVariance);

    if (verbose_) {
        std::cout << "initialized BF relevances" << std::endl;
    }
}

void GPz::buildMissingCache_(const Mat2d& input) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // Cache system to save data related to each combination of missing bands
    // encountered in the training data

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
        int bestID = 0;
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

void GPz::addMatrixElements_(const Mat2d& in, Mat2d& out, const MissingCacheElement& element,
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

    std::vector<Mat2d> isigma(m);
    std::vector<Mat2d> sigma(m);
    for (uint_t i = 0; i < m; ++i) {
        const Mat2d& fullGamma = parameters_.basisFunctionCovariances[i];
        isigma[i] = fullGamma.transpose()*fullGamma;
        sigma[i] = computeInverseSymmetric(isigma[i]);
    }

    for (auto& cacheItem : missingCache_) {
        cacheItem.covariancesObserved.resize(m);
        cacheItem.invCovariancesObserved.resize(m);
        cacheItem.covariancesObservedLogDeterminant.resize(m);
        if (what == MissingCacheUpdate::TRAIN) {
            cacheItem.gUO.resize(m);
            cacheItem.dgO.resize(m);
        } else if (what == MissingCacheUpdate::PREDICT) {
            cacheItem.R.resize(m);
            cacheItem.Psi_hat.resize(m);
        }

        for (uint_t i = 0; i < m; ++i) {
            const Mat2d& fullGamma = parameters_.basisFunctionCovariances[i]; // GPzMatLab: Gamma(:,:,i)

            // Fetch elements of the covariance matrix for non-missing bands
            fetchMatrixElements_(cacheItem.covariancesObserved[i], sigma[i], cacheItem, 'o', 'o');

            // Compute inverse
            cacheItem.invCovariancesObserved[i] = computeInverseSymmetric(cacheItem.covariancesObserved[i]);

            // Compute log determinant
            cacheItem.covariancesObservedLogDeterminant[i] = computeLogDeterminant(cacheItem.covariancesObserved[i]);

            if (what == MissingCacheUpdate::TRAIN) {
                if (cacheItem.countMissing != 0) {
                    // Compute gUO and dgO (for training)
                    Mat2d isigmaMissing;  // GPzMatLab: iSigma(u,u)
                    Mat2d isigmaObserved; // GPzMatLab: iSigma(u,o)
                    fetchMatrixElements_(isigmaMissing,  isigma[i], cacheItem, 'u', 'u');
                    fetchMatrixElements_(isigmaObserved, isigma[i], cacheItem, 'u', 'o');

                    Mat2d gammaMissing;  // GPzMatLab: Gamma(:,u,i)
                    Mat2d gammaObserved; // GPzMatLab: Gamma(:,o,i)
                    fetchMatrixElements_(gammaMissing,   fullGamma, cacheItem, ':', 'u');
                    fetchMatrixElements_(gammaObserved,  fullGamma, cacheItem, ':', 'o');

                    cacheItem.gUO[i] = computeInverseSymmetric(isigmaMissing)*isigmaObserved;
                    cacheItem.dgO[i] = 2*(gammaObserved - gammaMissing*cacheItem.gUO[i]);
                } else {
                    cacheItem.gUO[i].resize(0,0);
                    cacheItem.dgO[i] = 2*fullGamma;
                }
            }

            if (what == MissingCacheUpdate::PREDICT) {
                if (cacheItem.countMissing != 0) {
                    // Compute R and Psi_hat (for prediction)
                    Mat2d sigmaMissing;  // GPzMatLab: Sigma(u,u)
                    Mat2d sigmaObserved; // GPzMatLab: Sigma(u,o)
                    fetchMatrixElements_(sigmaMissing,  sigma[i], cacheItem, 'u', 'u');
                    fetchMatrixElements_(sigmaObserved, sigma[i], cacheItem, 'u', 'o');

                    Eigen::LDLT<Mat2d> cholesky(sigmaObserved);
                    cacheItem.R[i] = cholesky.solve(sigmaMissing.transpose());
                    cacheItem.Psi_hat[i] = sigmaMissing - sigmaObserved*cacheItem.R[i];
                } else {
                    cacheItem.R[i].resize(0,0);
                    cacheItem.Psi_hat[i].resize(0,0);
                }
            }
        }
    }
}

void GPz::updateTrainBasisFunctions_() {
    trainBasisFunctions_ = evaluateBasisFunctions_(inputTrain_, inputErrorTrain_, missingTrain_);
}

void GPz::updateValidBasisFunctions_() {
    validBasisFunctions_ = evaluateBasisFunctions_(inputValid_, inputErrorValid_, missingValid_);
}

void GPz::updateTrainOutputErrors_() {
    trainOutputLogError_ = evaluateOutputLogErrors_(trainBasisFunctions_);
}

void GPz::updateValidOutputErrors_() {
    validOutputLogError_ = evaluateOutputLogErrors_(validBasisFunctions_);
}

Mat1d GPz::evaluateBasisFunctions_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    Eigen::JacobiSVD<Mat2d> svd;

    Mat1d funcs(m);
    for (uint_t j = 0; j < m; ++j) {
        Mat1d delta = input - parameters_.basisFunctionPositions.row(j).transpose(); // GPzMatLab: Delta(i,:)

        double value = log(2.0)*element.countMissing;

        Mat2d invCovariance; // GPzMatLab: inv(Sigma(o,o)) or inv(PsiPlusSigma)

        if (inputError.rows() == 0) {
            invCovariance = element.invCovariancesObserved[j];
        } else {
            Mat1d varianceObserved;
            fetchVectorElements_(varianceObserved, inputError, element, 'o');

            Mat2d covariance = element.covariancesObserved[j]; // GPzMatLab: PsiPlusSigma
            for (uint_t k = 0; k < d - element.countMissing; ++k) {
                covariance(k,k) += varianceObserved[k];
            }

            svd.compute(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
            invCovariance = computeInverseSymmetric(covariance, svd);
            value += computeLogDeterminant(svd) - element.covariancesObservedLogDeterminant[j];
        }

        for (uint_t k = 0; k < d; ++k)
        for (uint_t l = k; l < d; ++l) {
            value += (l == k ? 1.0 : 2.0)*invCovariance(k,l)*delta[k]*delta[l];
        }

        funcs[j] = exp(-0.5*value);
    }

    return funcs;
}

Mat2d GPz::evaluateBasisFunctions_(const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const {
    const uint_t n = input.rows();
    const uint_t m = numberBasisFunctions_;

    Mat2d funcs(n,m);
    for (uint_t i = 0; i < n; ++i) {
        const MissingCacheElement& element = getMissingCacheElement_(missing[i]);

        if (inputError.rows() == 0) {
            funcs.row(i) = evaluateBasisFunctions_(input.row(i), Mat1d{}, element).transpose();
        } else {
            funcs.row(i) = evaluateBasisFunctions_(input.row(i), inputError.row(i), element).transpose();
        }
    }

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
    const uint_t m = numberBasisFunctions_;

    const bool updateLikelihood =
        requested == Minimize::FunctionOutput::ALL_TRAIN ||
        requested == Minimize::FunctionOutput::METRIC_TRAIN;

    const bool updateDerivatives =
        requested == Minimize::FunctionOutput::ALL_TRAIN ||
        requested == Minimize::FunctionOutput::DERIVATIVES_TRAIN;

    if (updateLikelihood) {
        logLikelihood_ = 0.0;
    }

    // Pre-compute things
    updateMissingCache_(MissingCacheUpdate::TRAIN);
    updateTrainBasisFunctions_();
    updateTrainOutputErrors_();

    // Do the hard work...
    Mat1d relevances = parameters_.basisFunctionLogRelevances.array().exp().matrix(); // GPzMatLab: alpha
    Mat1d errorRelevances;     // GPzMatLab: tau
    Mat1d errorWeightsSquared; // GPzMatLab: v.^2
    if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
        errorRelevances = parameters_.uncertaintyBasisLogRelevances.array().exp().matrix();
        errorWeightsSquared = parameters_.uncertaintyBasisWeights.array().pow(2).matrix();
    }

    Vec1d trainOutputError = (-trainOutputLogError_).array().exp(); // GPzMatLab: beta
    Vec1d dataWeight = weightTrain_*trainOutputError; // GPzMatLab: omega_x_beta
    Mat2d weightedBasisFunctions = trainBasisFunctions_; // GPzMatLab: BxPHI
    for (uint_t i = 0; i < n; ++i) {
        weightedBasisFunctions.row(i) *= dataWeight[i];
    }

    Mat2d modelCovariance = weightedBasisFunctions.transpose()*trainBasisFunctions_;
    modelCovariance.diagonal() += relevances; // GPzMatLab: SIGMA

    Eigen::JacobiSVD<Mat2d> svd(modelCovariance, Eigen::ComputeThinU | Eigen::ComputeThinV);

    modelInvCovariance_ = computeInverseSymmetric(modelCovariance, svd);
    modelWeights_ = modelInvCovariance_*weightedBasisFunctions.transpose()*outputTrain_.matrix();

    Mat1d deviates = trainBasisFunctions_*modelWeights_ - outputTrain_; // GPzMatLab: delta
    Mat1d weightedDeviates = (dataWeight.array()*deviates.array()).matrix(); // GpzMatLab: omega_beta_x_delta

    if (updateLikelihood) {
        // Log likelihood
        // ==============

        logLikelihood_ = -0.5*(weightedDeviates.array()*deviates.array()).sum()
            -0.5*(relevances.array()*modelWeights_.array().pow(2)).sum()
            +0.5*parameters_.basisFunctionLogRelevances.sum()
            -0.5*computeLogDeterminant(svd)
            +0.5*((-trainOutputLogError_).array()*weightTrain_.array()).sum()
            -0.5*log(2.0*M_PI)*sumWeightTrain_;

        if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
            logLikelihood_ += -0.5*(errorWeightsSquared.array()*errorRelevances.array()).sum()
                +0.5*parameters_.uncertaintyBasisLogRelevances.sum()
                -0.5*m*log(2.0*M_PI);
        }
    }

    if (updateDerivatives) {
        // Derivative of basis functions
        // =============================

        Mat2d derivBasis = -weightedBasisFunctions*modelInvCovariance_
            -weightedDeviates*modelWeights_.transpose(); // GPzMatLab: dlnPHI

        // Derivative wrt relevance
        // ========================

        Mat1d dwda = -modelInvCovariance_*(relevances.array()*modelWeights_.array()).matrix();

        derivatives_.basisFunctionLogRelevances = 0.5
            -0.5*modelInvCovariance_.diagonal().array()*relevances.array()
            -(trainBasisFunctions_.transpose()*weightedDeviates).array()*dwda.array()
            -relevances.array()*modelWeights_.array()*dwda.array()
            -0.5*relevances.array()*modelWeights_.array().pow(2);

        // Derivative wrt uncertainty constant
        // ===================================

        Mat1d nu = ((trainBasisFunctions_*modelInvCovariance_).array()*trainBasisFunctions_.array())
            .rowwise().sum(); // GPzMatLab: nu

        Mat1d derivOutputError(n); // GPzMatLab: dbeta
        // Do this in an explicit loop as all operations are component-wise, it's clearer
        for (uint_t i = 0; i < n; ++i) {
            derivOutputError[i] = -0.5*weightTrain_[i]*
                (1.0 - trainOutputError[i]*(deviates[i]*deviates[i] + nu[i]));
        }

        derivatives_.logUncertaintyConstant = derivOutputError.sum();

        if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
            // Derivative wrt uncertainty weights
            // ==================================

            Mat1d weightedRelevance = parameters_.uncertaintyBasisWeights.array()*errorRelevances.array();

            derivatives_.uncertaintyBasisWeights = trainBasisFunctions_.transpose()*derivOutputError
                -weightedRelevance;

            // Derivative wrt uncertainty relevances
            // =====================================

            derivatives_.uncertaintyBasisLogRelevances = 0.5
            -0-0.5*weightedRelevance.array()*errorRelevances.array();

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

        for (uint_t i = 0; i < n; ++i) {
            const MissingCacheElement& element = getMissingCacheElement_(missingTrain_[i]);

            for (uint_t j = 0; j < m; ++j) {
                Mat1d delta = inputTrain_.row(i) - parameters_.basisFunctionPositions.row(j); // GPzMatLab: Delta(i,:)

                Mat2d invCovariance;      // GPzMatLab: inv(Sigma(o,o)) or iPSoo
                Mat2d derivInvCovariance; // GPzMatLab: diSoo

                if (inputErrorTrain_.rows() == 0) {
                    invCovariance = element.invCovariancesObserved[j];
                    derivInvCovariance = ((-0.5*derivBasis(i,j))*delta)*delta.transpose();
                } else {
                    Mat2d variance = inputErrorTrain_.row(i).asDiagonal(); // GPzMatLab: Psi(:,:,i)
                    Mat2d varianceObserved; // GPzMatLab: Psi(o,o,i)
                    fetchMatrixElements_(varianceObserved, variance, element, 'o', 'o');

                    Mat2d covariance = element.covariancesObserved[j] + varianceObserved;
                    invCovariance = computeInverseSymmetric(covariance); // GPzMatLab: iPSoo

                    Mat2d derivCovariance = 0.5*(element.invCovariancesObserved[j] - invCovariance
                        + invCovariance*(delta*delta.transpose())*invCovariance); // GPzMatLab: dSoo

                    derivInvCovariance = (-derivBasis(i,j))*element.covariancesObserved[j]*derivCovariance*element.covariancesObserved[j];
                }

                // Derivative wrt to basis positions
                // =================================
                derivatives_.basisFunctionPositions.row(j) += derivBasis(i,j)*delta.transpose()*invCovariance;

                // Derivative wrt to basis covariances
                // =================================
                Mat2d dgO = element.dgO[j]*derivInvCovariance;
                addMatrixElements_(dgO, derivatives_.basisFunctionCovariances[j], element, ':', 'o');

                if (element.countMissing != 0) {
                    Mat2d dgU = -dgO*element.gUO[j].transpose();
                    addMatrixElements_(dgU, derivatives_.basisFunctionCovariances[j], element, ':', 'u');
                }
            }
        }
    }
}

void GPz::updateLikelihoodValid_() {
    const uint_t n = inputValid_.rows();

    // Pre-compute things
    updateValidBasisFunctions_();
    updateValidOutputErrors_();

    // Do the hard work... (simple Gaussian likelihood (model-obs)/errors ~ chi2)
    Mat1d deviates = validBasisFunctions_*modelWeights_ - outputValid_; // GPzMatLab: delta

    logLikelihoodValid_ = 0.0;
    for (uint_t i = 0; i < n; ++i) {
        logLikelihoodValid_ -= weightValid_[i]*0.5*
            (-validOutputLogError_[i] + exp(-validOutputLogError_[i])*deviates[i]*deviates[i]);
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

    // No missing data, but we have noisy inputs

    Mat1d basis = evaluateBasisFunctions_(input, Mat1d{}, element);

    value = basis.transpose()*modelWeights_; // GPzMatLab: mu

    varianceTrainDensity = 0.0; // GPzMatLab: nu
    varianceInputNoise = 0.0; // GPzMatLab: gamma
    double VlnS = 0.0;

    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j <= i; ++j) {
        Mat2d iCij = element.invCovariancesObserved[i] + element.invCovariancesObserved[j];

        Eigen::JacobiSVD<Mat2d> svd(iCij, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Mat2d Cij = computeInverseSymmetric(iCij, svd);
        Mat1d cij = parameters_.basisFunctionPositions.row(i)*element.invCovariancesObserved[i]
                  + parameters_.basisFunctionPositions.row(j)*element.invCovariancesObserved[j];

        cij = svd.solve(cij);

        Mat1d Delta = parameters_.basisFunctionPositions.row(i)
                    - parameters_.basisFunctionPositions.row(j);

        // Now this is source-specific

        Delta = input - cij;
        Cij.diagonal() += inputError;
        svd.compute(Cij);

        double lnNxc = -0.5*svd.solve(Delta).transpose()*Delta - 0.5*computeLogDeterminant(svd);
        double ZijNxc = exp(lnZ(i,j) + lnNxc);

        double coef = (i == j ? 1.0 : 2.0)*ZijNxc;
        varianceTrainDensity += coef*modelInvCovariance_(i,j);
        varianceInputNoise += coef*modelWeights_[i]*modelWeights_[j];
        VlnS += coef*parameters_.uncertaintyBasisWeights[i]*parameters_.uncertaintyBasisWeights[j];
    }

    varianceInputNoise -= value*value;

    double logError = evaluateOutputLogError_(basis);
    VlnS -= pow(logError - parameters_.logUncertaintyConstant, 2);
    varianceTrainNoise = exp(logError)*(1.0 + 0.5*VlnS); // GPzMatLab: beta_i
}

void GPz::predictMissingNoisy_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element, double& value,
    double& varianceTrainDensity, double& varianceTrainNoise, double& varianceInputNoise) const {

    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;
    const uint_t nu = element.countMissing;
    const uint_t no = d - nu;
    const bool noError = inputError.rows() == 0;

    Mat2d filledInput(d,m); // GPzMatLab: X_hat
    Mat1d Pio(m); // GPzMatLab: Ex & Pio (combined)
    std::vector<Mat2d> Psi_hat(m); // GPzMatLab: Psi_hat

    for (uint_t i = 0; i < m; ++i) {
        Mat2d sigma = element.covariancesObserved[i];
        if (!noError) {
            sigma.diagonal() += inputError;
        }

        Eigen::JacobiSVD<Mat2d> svd(sigma, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Mat1d position = parameters_.basisFunctionPositions.row(i);
        Mat1d Delta = input - position;
        Mat1d DeltaObserved;
        fetchVectorElements_(DeltaObserved, Delta, element, 'o');

        Pio[i] = -0.5*svd.solve(DeltaObserved).transpose()*DeltaObserved
            -0.5*computeLogDeterminant(svd);
        Pio[i] = exp(Pio[i])*modelInputPrior_[i];

        Mat1d positionMissing; // GPzMatLab: P(i,~o)
        fetchVectorElements_(positionMissing, position, element, 'u');

        Mat1d inputNew = element.R[i]*DeltaObserved + positionMissing;

        uint_t k = 0;
        for (uint_t j = 0; j < d; ++j) {
            if (element.missing[j]) {
                filledInput(j,i) = inputNew[k];
                ++k;
            } else {
                filledInput(j,i) = input[j];
            }
        }

        if (noError) {
            Psi_hat[i].resize(d,d);
        } else {
            Mat1d errorObserved;
            fetchVectorElements_(errorObserved, inputError, element, 'o');
            Mat2d t(d, no);
            t.block(0,  0, no, no) = Mat2d::Identity(no, no);
            t.block(no, 0, nu, no) = element.R[i].transpose();

            Psi_hat[i] = t*errorObserved.asDiagonal()*t.transpose();
        }

        addMatrixElements_(element.Psi_hat[i], Psi_hat[i], element, 'u', 'u');
    }

    Pio /= Pio.sum();

    Mat1d basis(m); // GPzMatLab: PHI

    varianceTrainDensity = 0.0; // GPzMatLab: nu
    varianceInputNoise = 0.0; // GPzMatLab: gamma
    double VlnS = 0.0;

    for (uint_t i = 0; i < m; ++i)
    for (uint_t j = 0; j <= i; ++j) {
        Mat2d iCij = element.invCovariancesObserved[i] + element.invCovariancesObserved[j];

        Eigen::JacobiSVD<Mat2d> svd(iCij, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Mat2d Cij = computeInverseSymmetric(iCij, svd);
        Mat1d cij = element.invCovariancesObserved[i]*parameters_.basisFunctionPositions.row(i)
                  + element.invCovariancesObserved[j]*parameters_.basisFunctionPositions.row(j);

        cij = svd.solve(cij);

        Mat1d Delta = filledInput.col(j) - parameters_.basisFunctionPositions.row(i).transpose();
        Mat2d Sij = noMissingCache_->covariancesObserved[i] + Psi_hat[j];
        svd.compute(Sij);
        double N = exp(-0.5*svd.solve(Delta).transpose()*Delta
                -0.5*computeLogDeterminant(svd));

        basis[i] += (i == j ? 1.0 : 2.0)*N*Pio[j];

        Delta = filledInput.col(i) - parameters_.basisFunctionPositions.row(j).transpose();
        Sij = noMissingCache_->covariancesObserved[j] + Psi_hat[i];
        svd.compute(Sij);
        N = exp(-0.5*svd.solve(Delta).transpose()*Delta
                -0.5*computeLogDeterminant(svd));

        basis[j] += (i == j ? 1.0 : 2.0)*N*Pio[i];

        double EcCij = 0.0;
        for (uint_t l = 0; l < m; ++l) {
            Delta = filledInput.col(l) - cij;
            Sij = Cij + Psi_hat[l];
            svd.compute(Sij);
            N = exp(-0.5*svd.solve(Delta).transpose()*Delta
                    -0.5*computeLogDeterminant(svd));

            EcCij += N*Pio[l];
        }

        double Zij = exp(lnZ(i,j))*EcCij;

        double coef = (i == j ? 1.0 : 2.0)*Zij;
        varianceTrainDensity += coef*modelInvCovariance_(i,j);
        varianceInputNoise += coef*modelWeights_[i]*modelWeights_[j];
        VlnS += coef*parameters_.uncertaintyBasisWeights[i]*parameters_.uncertaintyBasisWeights[j];
    }

    basis.array() *= exp(noMissingCache_->covariancesObservedLogDeterminant);

    value = basis.transpose()*modelWeights_; // GPzMatLab: mu

    varianceInputNoise -= value*value;

    double logError = evaluateOutputLogError_(basis);
    VlnS -= pow(logError - parameters_.logUncertaintyConstant, 2);
    varianceTrainNoise = exp(logError)*(1.0 + 0.5*VlnS); // GPzMatLab: beta_i
}

GPzOutput GPz::predict_(const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const {
    const uint_t n = input.rows();
    const uint_t m = numberBasisFunctions_;
    const bool noError = inputError.rows() == 0;

    GPzOutput result;
    result.value.resize(n);
    result.varianceTrainDensity.resize(n);
    result.varianceTrainNoise.resize(n);
    result.varianceInputNoise.resize(n);

    // Compute cached variables
    lnZ.resize(m,m); {
        for (uint_t i = 0; i < m; ++i)
        for (uint_t j = 0; j <= i; ++j) {
            Mat2d Sij = noMissingCache_->covariancesObserved[i]
                +noMissingCache_->covariancesObserved[j];

            Eigen::JacobiSVD<Mat2d> svd(Sij, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Mat1d Delta = parameters_.basisFunctionPositions.row(i)
                        - parameters_.basisFunctionPositions.row(j);

            lnZ(i,j) = -0.5*noMissingCache_->covariancesObservedLogDeterminant[i]
                -0.5*noMissingCache_->covariancesObservedLogDeterminant[j]
                -0.5*svd.solve(Delta).transpose()*Delta
                -0.5*computeLogDeterminant(svd);
        }
    }

    // Iterate over galaxies
    for (uint_t i = 0; i < n; ++i) {
        const MissingCacheElement& element = getMissingCacheElement_(missing[i]);

        if (element.countMissing == 0) {
            if (noError) {
                predictFull_(input.row(i), element, result.value[i], result.varianceTrainDensity[i],
                    result.varianceTrainNoise[i]);
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
            } else {
                predictMissingNoisy_(input.row(i), inputError.row(i), element, result.value[i],
                    result.varianceTrainDensity[i], result.varianceTrainNoise[i],
                    result.varianceInputNoise[i]);
            }
        }
    }

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
        assert(newFunction != PriorMeanFunction::LINEAR_MARGINALIZE && "not implemented");
        assert(newFunction != PriorMeanFunction::LINEAR_PREPROCESS  && "not implemented");

        priorMean_ = newFunction;
        reset_();
    }
}

PriorMeanFunction GPz::getPriorMeanFunction() const {
    return priorMean_;
}

void GPz::setNumberOfFeatures(uint_t num) {
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

// =====================
// Fit/training function
// =====================

void GPz::fit(Mat2d input, Mat2d inputError, Vec1d output) {
    // Check inputs are consistent
    assert(checkErrorDimensions_(input, inputError) && "input uncertainty has incorrect dimension");

    if (verbose_) {
        std::cout << "begin fitting with " << input.rows() << " data points" << std::endl;
        std::cout << "found " << input.cols() << " features" << std::endl;
        if (inputError.rows() != 0) {
            std::cout << "found uncertainties for the features" << std::endl;
        }
    }

    // Normalize the inputs
    setNumberOfFeatures(input.cols());
    initializeInputs_(std::move(input), std::move(inputError), std::move(output));

    // Setup the fit, initialize arrays, etc.
    initializeFit_();

    // Build vector with initial values for hyper-parameter
    Vec1d initialValues = makeParameterArray_(parameters_);

    if (verbose_) {
        std::cout << "starting optimization of model" << std::endl;
    }

    // Use BFGS for minimization
    Minimize::Options options;
    options.maxIterations = optimizationMaxIterations_;
    options.hasValidation = inputValid_.rows() != 0;
    options.minimizerTolerance = optimizationTolerance_;
    options.gradientTolerance = optimizationGradientTolerance_;
    options.verbose = verbose_;

    Minimize::Result result = Minimize::minimizeBFGS(options, initialValues,
        // Minimization function
        [this](const Vec1d& vectorParameters, Minimize::FunctionOutput requested) {

            if (requested == Minimize::FunctionOutput::METRIC_VALID) {
                // Compute likelihood of the validation set
                updateLikelihoodValid_();

                // Return only the log likelihood
                Vec1d result(1);
                result[0] = -logLikelihoodValid_/inputValid_.rows();

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
        }
    );

    assert(result.success && "minimization failed");

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

GPzOutput GPz::predict(Mat2d input, Mat2d inputError) const {
    // Check input is consistent
    assert(checkInputDimensions_(input) && "input has incorrect dimension");
    assert(checkErrorDimensions_(input, inputError) && "input uncertainty has incorrect dimension");

    // Check that we have a usable set of parameters to make predictions
    assert(parameters_.basisFunctionPositions.rows() != 0 && "model is not initialized");

    // Detect missing data
    buildMissingCache_(input);
    updateMissingCache_(MissingCacheUpdate::PREDICT);

    Vec1i missing = getBestMissingID_(input);

    // Project input from real space to training space
    applyInputNormalization_(input, inputError);

    // Make prediction
    GPzOutput result = predict_(input, inputError, missing);

    // De-project output from training space to real space
    restoreOutputNormalization_(input, missing, result);

    return result;
}

}  // namespace PHZ_GPz


