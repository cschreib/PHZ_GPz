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
#include <random>
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
        outputParams[ip] = inputParams.basisFunctionRelevances[i];
        ++ip
    }

    // TODO: include prior mean function

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

    // TODO: include prior mean function

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

    assert(ip == inputParams.size());
}

void GPz::resizeHyperParameters_(HyperParameters& params) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;

    // TODO: include prior mean function

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

    // TODO: include prior mean function

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
}

void GPz::reset_() {
    parameters_ = HyperParameters{};
    derivatives_ = HyperParameters{};

    featureMean_.clear();
    featureSigma_.clear();
    outputMean_ = 0.0;

    featurePCAMean_.clear();
    featurePCASigma_.clear();
    featurePCABasisVectors_.clear();
}

void GPz::applyInputNormalization_(Mat2d& input, Mat2d& inputError) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    if (normalizationScheme_ == NormalizationScheme::WHITEN) {
        for (uint_t i = 0; i < n; ++i)
        for (uint_t j = 0; j < d; ++j) {
            input(i,j) = (input(i,j) - featureMean_[j])/featureSigma_[j];
        }
    }

    // TODO: normalize error too?
}

void GPz::applyOutputNormalization_(Vec1d& output) const {
    output -= outputMean_;
}

void GPz::restoreOutputNormalization_(Vec1d& output) const {
    output += outputMean_;
}

void GPz::normalizeInputs_(Mat2d& input, Mat2d& inputError, Vec1d& output) {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    if (normalizationScheme_ == NormalizationScheme::WHITEN) {
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

    // Compute output mean
    outputMean_ = output.mean();

    applyInputNormalization_(input, inputError_);
    applyOutputNormalization_(output);

    // TODO: implement fixPsi() for error (see if it is applied to prediciton input too)
}

void GPz::splitTrainValid_(const Mat2d& input, const Mat2d& inputError, const Vec1d& output) {
    if (trainValidRatio_ == 1.0) {
        // No validation set
        inputTrain_ = input;
        inputErrorTrain_ = inputError;
        outputTrain_ = output;

        inputValid_.clear();
        inputErrorValid_.clear();
        outputValid_.clear();
    } else {
        // Randomly shuffle the data
        std::mt19937 seed(seedTrainSplit_);
        std::vector<uint_t> indices(input.cols);
        std::iota(indices.begin(), indices.end(), 0u);
        std::shuffle(indices.begin(), indices.end(), seed);

        uint_t numberTrain = round(input.rows()*trainValidRatio_);
        assert(numberTrain != 0);

        uint_t numberValid = input.rows() - numberTrain;

        // Pick training data from first part of shuffled data
        inputTrain_.resize(numberTrain, numberFeatures_);
        if (!inputError.empty()) {
            inputTrainError_.resize(numberTrain, numberFeatures_);
        } else {
            inputTrainError_.clear();
        }
        outputTrain_.resize(numberTrain);

        for (uint_t i = 0; i < numberTrain; ++i) {
            uint_t index = indices[i];
            outputTrain_[i] = output[index];
            for (uint_t j = 0; j < numberFeatures_; ++j) {
                inputTrain_(i,j) = input(index,j);
                if (!inputError.empty()) {
                    inputErrorTrain_(i,j) = inputError(index,j);
                }
            }
        }

        // Pick validation data from second part of shuffled data
        inputValid_.resize(numberValid, numberFeatures_);
        if (inputError.empty()) {
            inputValidError_.resize(numberValid, numberFeatures_);
        } else {
            inputValidError_.clear();
        }
        outputValid_.resize(numberValid);

        for (uint_t i = 0; i < numberValid; ++i) {
            uint_t index = indices[i + numberTrain];
            outputValid_[i] = output[index];
            for (uint_t j = 0; j < numberFeatures_; ++j) {
                inputValid_(i,j) = input(index,j);
                if (!inputError.empty()) {
                    inputErrorValid_(i,j) = inputError(index,j);
                }
            }
        }
    }
}

void GPz::initializeInputs_(Mat2d input, Mat2d inputError, Vec1d output) {
    normalizeInputs_(input, inputError, output);
    splitTrainValid_(input, inputError, output);
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
    assert(solver.info() == Eigen::Success);

    Vec1d eigenValues = solver.eigenvalues().real();
    Mat2d eigenVectors = solver.eigenvectors().real();

    eigenValues = sqrt(eigenValues/(n-1.0));
    featurePCABasisVectors_ = eigenValues.asDiagonal()*eigenVectors.transpose();

    featurePCASigma_ /= n;
}

void GPz::initializeBasisFunctions_() {
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
        parameters_.basisFunctionPositions.row(i) += featurePCAMean_;
    }
}

void GPz::initializeBasisFunctionRelevances_() {
    double outputLogVariance = log(outputTrain_.square().sum()/(n-1.0));
    parameters_.basisFunctionRelevances.fill(-outputLogVariance);
}

Mat2d GPz::initializeCovariancesFillLinear_(Mat2d input) const {
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    // TODO: placeholder

    return input;
}

Vec1d GPz::initializeCovariancesMakeGamma_(const Mat2d& input) const {
    const uint_t m = numberBasisFunctions_;
    const uint_t d = numberFeatures_;
    const uint_t n = input.rows();

    Mat2d linearInputs = initializeCovariancesFillLinear_(input);

    Vec1d gamma(m);
    double factor = 0.5*pow(m, 1.0/d);
    for (uint_t i = 0; i < m; ++i) {
        // double meanSquaredDist = 0.0;
        // for (uint_t j = 0; j < n; ++j)
        // for (uint_t k = 0; k < d; ++k) {
        //     double d = parameters_.basisFunctionPositions(i,k) - linearInputs(j,k);
        //     me += d*d;
        // }

        double meanSquaredDist = 0.0;
        for (uint_t j = 0; j < n; ++j) {
            meanSquaredDist +=
                (parameters_.basisFunctionPositions.row(i) - linearInputs.row(j)).square().sum();
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
    Vec1d gamma = initializeCovariancesMakeGamma_(inputTrain_);

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
    const uint_t m = numberBasisFunctions_;
    const uint_t n = inputTrain_.rows();

    double outputLogVariance = log(outputTrain_.square().sum()/(n-1.0));
    parameters_.uncertaintyConstant = outputLogVariance;

    if (outputUncertaintyType_ == OutputUncertaintyType::INPUT_DEPENDENT) {
        for (uint_t i = 0; i < m; ++i) {
            parameters_.uncertaintyBasisWeights[i] = 0.0;
            parameters_.uncertaintyBasisRelevances[i] = 0.0;
        }
    }
}

void GPz::initializeFit_() {
    // Create arrays, matrices, etc.
    setNumberOfFeatures(inputTrain_.cols());
    updateNumberParameters_();
    resizeArrays_();

    // Pre-compute some things
    computeTrainingPCA_();

    // Set initial values for hyper-parameters
    initializeBasisFunctions_();
    initializeBasisFunctionRelevances_();
    initializeCovariances_();
    initializeErrors_();
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

Vec1d GPz::predict_(const Mat2d& input, const Mat2d& inputError) {
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

void GPz::setNormalizationScheme(NormalizationScheme scheme) {
    normalizationScheme_ = scheme;
}

NormalizationScheme GPz::getNormalizationScheme() const {
    return normalizationScheme_;
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

uint_t getTrainValidationSplitSeed() const {
    return seedTrainSplit_;
}

void setInitialPositionSeed(uint_t seed) {
    seedPositions_ = seed;
}

uint_t getInitialPositionSeed() const {
    return seedPositions_;
}

// =====================
// Fit/training function
// =====================

void GPz::fit(Mat2d input, Mat2d inputError, Vec1d output) {
    // Check inputs are consistent
    assert(inputError.empty() ||
        (inputError.rows() == input.rows() && inputError.cols() == input.cols()));

    // Normalize the inputs
    initializeInputs_(std::move(input), std::move(inputError), std::move(output));

    // Setup the fit, initialize arrays, etc.
    initializeFit_();

    // Build vector with initial values for hyper-parameter
    Vec1d initialValues = makeParameterArray_(parameters_);

    // Use BFGS for minimization
    Minimize::Options options;
    options.hasValidation = !inputValid_.empty();
    // TODO: tweak the BFGS options to copy behavior of original GPz

    Minimize::minimizeBFGS(options, initialValues,
        [this](const Vec1d& vectorParameters, Minimize::FunctionOutput requested) {

            Vec1d result;

            if (requested == Minimize::FunctionOutput::METRIC_VALID) {
                result.resize(1);
            } else {
                result.resize(1+numberParameters_);

                // Load new parameters
                loadParametersArray_(vectorParameters, parameters_);
            }

            // Compute/update the requested quantities
            updateLikelihood_(requested);

            if (requested == Minimize::FunctionOutput::METRIC_VALID) {
                // Return only the log likelihood of the validation set
                result[0] = logLikelihoodValid_;
            } else {
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

Vec1d GPz::predict(Mat2d input, Mat2d inputError) const {
    // Check input is consistent
    assert(input.cols() == numberFeatures_);
    assert(inputError.empty() ||
        (inputError.rows() == input.rows() && inputError.cols() == input.cols()));

    // Check that we have a usable set of parameters to make predictions
    assert(!parameters_.basisFunctionPositions.empty());

    // Project input from real space to training space
    applyInputNormalization_(input, inputError);

    // Make prediction
    Vec1d output = predict_(input, inputError);

    // De-project output from training space to real space
    restoreOutputNormalization_(output);

    return output;
}

Vec1d GPz::predict(Mat2d input) const {
    return predict(st::move(input), Mat2d());
}

}  // namespace PHZ_GPz


