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

#include "PHZ_GPz/STLWrapper.h"
#include "PHZ_GPz/EigenWrapper.h"
#include "PHZ_GPz/GSLWrapper.h"
#include <vector>

namespace PHZ_GPz {

// ==============================
// Enumerations for configuration
// ==============================

/**
 * @brief Choice of prior mean function
 */
enum class PriorMeanFunction {
    ZERO, LINEAR
};

/**
 * @brief Choice of covariance parametrization
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
enum class OutputUncertaintyType {
    UNIFORM,
    INPUT_DEPENDENT
};

/** \var PHZ_GPz::OutputUncertaintyType::UNIFORM
 * @brief Uniform uncertainty
 *
 * The uncertainty on the output is assumed to be uniform, namely, it does not depend on the inputs.
 */

/** \var PHZ_GPz::OutputUncertaintyType::INPUT_DEPENDENT
 * @brief Input-dependent uncertainty
 *
 * The uncertainty on the output is modeled as a linear combination of basis functions,
 * with weights independent of those used to model the output value itself. This leads to an
 * input-dependent uncertainty (heteroscedastic noise). Selecting this model increases the number
 * of hyper-parameters by two times the number of basis functions.
 */

/**
 * @brief Choice of training data weighting scheme
 */
enum class WeightingScheme {
    NATURAL,
    ONE_OVER_ONE_PLUS_OUTPUT,
    BALANCED
};

/**
 * @brief Normalization of input data
 */
enum class NormalizationScheme {
    NATURAL,
    WHITEN
};

// =========
// GPz class
// =========

/**
 * @class GPz
 * @brief Training and prediction using Gaussian Processes
 *
 */
class GPz {

    // =======================
    // Configuration variables
    // =======================

    uint_t                numberBasisFunctions_ = 100;
    PriorMeanFunction     priorMean_ = PriorMeanFunction::LINEAR;
    CovarianceType        covarianceType_ = CovarianceType::VARIABLE_COVARIANCE;
    OutputUncertaintyType outputUncertaintyType_ = OutputUncertaintyType::INPUT_DEPENDENT;
    WeightingScheme       weightingScheme_ = WeightingScheme::BALANCED;
    NormalizationScheme   normalizationScheme_ = NormalizationScheme::WHITEN;
    double                balancedWeightingBinSize_ = 0.1;
    double                trainValidRatio_ = 0.5;

    // ==================
    // Indexing variables
    // ==================

    uint_t numberFeatures_ = 0;
    uint_t numberParameters_ = 0;

    // NB: these are just used for debugging purposes

    uint_t indexBasisPosition_ = 0;
    uint_t indexBasisRelevance_ = 0;
    uint_t indexBasisCovariance_ = 0;
    uint_t indexError_ = 0;
    uint_t indexErrorWeight_ = 0;
    uint_t indexErrorRelevance_ = 0;
    uint_t indexPriorMean_ = 0;

    // ================
    // Hyper-parameters
    // ================

    struct HyperParameters {
        Mat2d              basisFunctionPositions;     // GPz MatLab: P
        Mat1d              basisFunctionRelevances;    // GPz MatLab: lnAlpha
        std::vector<Mat2d> basisFunctionCovariances;   // GPz MatLab: Gamma
        double             uncertaintyConstant = 0.0;  // GPz MatLab: b
        Mat1d              uncertaintyBasisWeights;    // GPz MatLab: v
        Mat1d              uncertaintyBasisRelevances; // GPz MatLab: lnTau
    };

    HyperParameters parameters_, derivatives_;

    // ==========================
    // Input projection variables
    // ==========================

    Vec1d  featureMean_;       // GPz MatLab: muX
    Vec1d  featureSigma_;      // GPz MatLab: sdX
    double outputMean_ = 0.0;  // GPz MatLab: muY

    Vec1d  featurePCAMean_;
    Vec1d  featurePCASigma_;
    Mat2d  featurePCABasisVectors_;

    // ===================
    // Randomization seeds
    // ===================

    uint_t seedTrainSplit_ = 42;
    uint_t seedPositions_ = 55;

    // ======================
    // Minimization variables
    // ======================

    Mat2d inputTrain_;      // GPzMatLab: X[training,:]
    Mat2d inputErrorTrain_; // GPzMatLab: Psi[training,:]
    Vec1d outputTrain_;     // GPzMatLab: Y[training,:]
    Vec1d weightTrain_;     // GPzMatLab: omega[training,:]
    Mat2d inputValid_;      // GPzMatLab: X[validation,:]
    Mat2d inputErrorValid_; // GPzMatLab: Psi[validation,:]
    Vec1d outputValid_;     // GPzMatLab: Y[validation,:]
    Vec1d weightValid_;     // GPzMatLab: omega[validation,:]

    double logLikelihood_ = 0.0;
    double logLikelihoodValid_ = 0.0;

    // ====================================
    // Internal functions: hyper-parameters
    // ====================================ip

    Vec1d makeParameterArray_(const HyperParameters& inputParams) const;

    void loadParametersArray_(const Vec1d& inputParams, HyperParameters& outputParams) const;

    void resizeHyperParameters_(HyperParameters& params) const;

    // ==================================
    // Internal functions: initialization
    // ==================================

    void updateNumberParameters_();

    void resizeArrays_();

    void reset_();

    void applyInputNormalization_(Mat2d& input, Mat2d& inputError) const;

    void applyOutputNormalization_(Vec1d& output) const;

    void restoreOutputNormalization_(Vec1d& output) const;

    void normalizeInputs_(Mat2d& input, Mat2d& inputError, Vec1d& output);

    void splitTrainValid_(const Mat2d& input, const Mat2d& inputError,
        const Vec1d& output, const Vec1d& weight);

    Vec1d computeWeights_(const Vec1d& output) const;

    void initializeInputs_(Mat2d input, Mat2d inputError, Vec1d output);

    void computeTrainingPCA_();

    void initializeBasisFunctions_();

    void initializeBasisFunctionRelevances_();

    Mat2d initializeCovariancesFillLinear_(Mat2d x) const;

    Vec1d initializeCovariancesMakeGamma_(const Mat2d& x) const;

    void initializeCovariances_();

    void initializeErrors_();

    void initializeFit_();

    // =======================
    // Internal functions: fit
    // =======================

    void updateLikelihood_(Minimize::FunctionOutput requested);

    // ==============================
    // Internal functions: prediction
    // ==============================

    Vec1d predict_(const Mat2d& x, const Mat2d& xe) const;

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

    void setNumberOfBasisFunctions(uint_t num);

    uint_t getNumberOfBasisFunctions() const;

    void setPriorMeanFunction(PriorMeanFunction newFunction);

    PriorMeanFunction getPriorMeanFunction() const;

    void setNumberOfFeatures(uint_t num);

    uint_t getNumberOfFeatures() const;

    void setWeightingScheme(WeightingScheme scheme);

    WeightingScheme getWeightingScheme() const;

    void setBalancedWeightingBinSize(double size);

    double getBalancedWeightingBinSize() const;

    void setNormalizationScheme(NormalizationScheme scheme);

    NormalizationScheme getNormalizationScheme() const;

    void setTrainValidationRatio(double ratio);

    double getTrainValidationRatio() const;

    void setTrainValidationSplitSeed(uint_t seed);

    uint_t getTrainValidationSplitSeed() const;

    void setInitialPositionSeed(uint_t seed);

    uint_t getInitialPositionSeed() const;

    // =====================
    // Fit/training function
    // =====================

    void fit(Mat2d input, Mat2d inputError, Vec1d output);

    // =================================
    // Fit/training result getter/setter
    // =================================

    Vec1d getParameters() const;

    void setParameters(const Vec1d& newParameters);

    // ===================
    // Prediction function
    // ===================

    Vec1d predict(Mat2d input, Mat2d inputError) const;

};  // End of GPz class

}  // namespace PHZ_GPz


#endif
