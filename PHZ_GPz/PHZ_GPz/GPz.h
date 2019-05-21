/**
 * @file PHZ_GPz/GPz.h
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
    ZERO,
    CONSTANT_PREPROCESS,
    LINEAR_PREPROCESS,
    LINEAR_MARGINALIZE
};

/** \var PHZ_GPz::PriorMeanFunction::ZERO
 * @brief Assume a prior of zero for data outside of the training coverage.
 *
 */

/** \var PHZ_GPz::PriorMeanFunction::CONSTANT_PREPROCESS
 * @brief Assume a constant prior for data outside of the training coverage (default).
 *
 * The prior is a constant, independent of the input values, computed as the mean of the data
 * as a pre-processing step before the training, and this value is subtracted from the target
 * outputs for training.
 */

/** \var PHZ_GPz::PriorMeanFunction::LINEAR_PREPROCESS
 * @brief Assume a linear prior for data outside of the training coverage (not yet implemented).
 *
 * The prior is a multi-linear function of the input values; the intercept and slope parameters
 * are fit to the data as a pre-processing step before the training, and the best-fit is subtracted
 * from the target outputs for training. This is less formally accurate than marginalizing over the
 * fit coefficients (LINEAR_MARGINALIZE).
 */

/** \var PHZ_GPz::PriorMeanFunction::LINEAR_MARGINALIZE
 * @brief Assume a linear prior for data outside of the training coverage (not yet implemented).
 *
 * The prior is a multi-linear function of the input values; the intercept and slope parameters
 * are marginalized over. This is implemented by introducing additional basis functions to the model
 * (one plus one per input feature), and allows the most faithfull predictions outside of the training
 * coverage.
 */

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

/** \var PHZ_GPz::CovarianceType::GLOBAL_LENGTH
 * @brief Model covariance with a uniform scale length (GPGL).
 *
 * Number of parameters is one.
 */

/** \var PHZ_GPz::CovarianceType::VARIABLE_LENGTH
 * @brief Model covariance with a scale length for each basis function (GPVL).
 *
 * Number of parameters is the number of basis functions.
 */

/** \var PHZ_GPz::CovarianceType::GLOBAL_DIAGONAL
 * @brief Model covariance with a uniform scale length for each feature (GPGD).
 *
 * Number of parameters is the number of features.
 */

/** \var PHZ_GPz::CovarianceType::VARIABLE_DIAGONAL
 * @brief Model covariance with a scale length for each feature and for each basis function (GPVD).
 *
 * Number of parameters is the number of basis functions times the number of features.
 */

/** \var PHZ_GPz::CovarianceType::GLOBAL_COVARIANCE
 * @brief Model covariance with a uniform covariance matrix (GPGC).
 *
 * Number of parameters is d*(d+1)/2 (where d is the number of features).
 */

/** \var PHZ_GPz::CovarianceType::VARIABLE_COVARIANCE
 * @brief Model covariance with a covariance matrix for each basis function (GPVC, default).
 *
 * Number of parameters is m*d*(d+1)/2 (where d is the number of features, and m is the number of
 * basis function).
 */

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
 * The uncertainty on the output is assumed to be uniform, namely, it is a single value
 * that does not depend on the inputs.
 */

/** \var PHZ_GPz::OutputUncertaintyType::INPUT_DEPENDENT
 * @brief Input-dependent uncertainty (default).
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
    UNIFORM,
    ONE_OVER_ONE_PLUS_OUTPUT,
    BALANCED
};

/** \var PHZ_GPz::WeightingScheme::UNIFORM
 * @brief All the training data has the same weight.
 */

/** \var PHZ_GPz::WeightingScheme::ONE_OVER_ONE_PLUS_OUTPUT
 * @brief Weight training data by 1/(1+z)^2 (where z is the output value).
 *
 * This weighting will minimize the standard photo-z metric Delta_z/(1+z).
 */

/** \var PHZ_GPz::WeightingScheme::BALANCED
 * @brief Weight training data uniformly in output space (default).
 *
 * This weights training data by '1/counts', where 'counts' is the number of training data points with
 * similar output value. The similarity criterion can be controlled using setBalancedWeightingBinSize().
 * This scheme avoids under-weighting the regions of the training set that are under-represented.
 */

/**
 * @brief Normalization of input data
 */
enum class NormalizationScheme {
    NATURAL,
    WHITEN
};

/** \var PHZ_GPz::NormalizationScheme::NATURAL
 * @brief The input data is not normalized.
 */

/** \var PHZ_GPz::NormalizationScheme::WHITEN
 * @brief The input data is whitened (default).
 *
 * Whitening is the process of substracting the data mean and dividing by the data standard deviation,
 * so that the data for each feature has a mean of zero and a standard deviation of unity. This allows
 * faster convergence of the hyper-parameters without loss of information.
 */

/**
 * @brief Method for splitting training and validation data
 */
enum class TrainValidationSplitMethod {
    RANDOM,
    SEQUENTIAL
};

/** \var PHZ_GPz::TrainValidationSplitMethod::RANDOM
 * @brief The training and validation data will be drawn at random from the input data (default).
 */

/** \var PHZ_GPz::TrainValidationSplitMethod::SEQUENTIAL
 * @brief The training data will be drawn from the first part of the input data,
 *        and validation from the second part.
 */

/**
 * @brief Choice of optimizer algorithm
 */
enum class OptimizerMethod {
    GSL_BFGS,
    GPZ_LBFGS
};

/** \var PHZ_GPz::OptimizerMethod::GSL_BFGS
 * @brief Use the BFGS algorithm provided by the GSL. For this optimizer, the recommended
 *        optimization tolerance threshold is 0.1.
 */

/** \var PHZ_GPz::OptimizerMethod::GPZ_LBFGS
 * @brief Use the custom L-BFGS aglorithm as implemented in GPz MatLab (default). For this
 *        optimizer, the recommended optimization tolerance threshold is 1e-9.
 */

/**
 * @struct GPzOutput
 * @brief Store the output of a GPz run
 *
 */
struct GPzOutput {
    Vec1d value;                /// Predicted value
    Vec1d variance;             /// Predicted total variance
    Vec1d varianceTrainDensity; /// Predicted variance due to density of training data
    Vec1d varianceTrainNoise;   /// Predicted variance due to training data noise
    Vec1d varianceInputNoise;   /// Predicted variance due to input noise
};

/**
 * @struct GPzHyperParameters
 * @brief Store the set of hyperparameters of a GPz run
 *
 */
struct GPzHyperParameters {
    Mat2d              basisFunctionPositions;        // GPz MatLab: P
    Mat1d              basisFunctionLogRelevances;    // GPz MatLab: lnAlpha
    std::vector<Mat2d> basisFunctionCovariances;      // GPz MatLab: Gamma
    double             logUncertaintyConstant = 0.0;  // GPz MatLab: b
    Mat1d              uncertaintyBasisWeights;       // GPz MatLab: v
    Mat1d              uncertaintyBasisLogRelevances; // GPz MatLab: lnTau
};

/**
 * @struct GPzModel
 * @brief Store the training state of a GPz run
 *
 */
struct GPzModel {
    // Projection of input/output space
    Vec1d featureMean;
    Vec1d featureSigma;
    double outputMean = 0.0;

    // Configuration
    PriorMeanFunction     priorMean = PriorMeanFunction::CONSTANT_PREPROCESS;
    OutputUncertaintyType outputUncertaintyType = OutputUncertaintyType::INPUT_DEPENDENT;
    NormalizationScheme   normalizationScheme = NormalizationScheme::WHITEN;

    // Fit hyper-parameters
    GPzHyperParameters parameters;

    // Fit parameters
    Vec1d modelWeights;
    Vec2d modelInvCovariance;
    Vec1d modelInputPrior;
};

/**
 * @struct GPzModel
 * @brief Store the optimization toggles for the GPz algorithm
 *
 */
struct GPzOptimizations {
    bool specializeForSingleFeature = true;
    bool specializeForDiagCovariance = true;
    bool enableMultithreading = true;
    uint_t maxThreads = 4;
};

/** \var PHZ_GPz::GPzOptimizations::specializeForSingleFeature
 * @brief Enable dedicated code for runs with a single feature.
 */

/** \var PHZ_GPz::GPzOptimizations::specializeForDiagCovariance
 * @brief Enable dedicated code for runs with diagonal covariances.
 */

/** \var PHZ_GPz::GPzOptimizations::enableMultithreading
 * @brief Enable parallel execution in some parts of the code.
 */

/** \var PHZ_GPz::GPzOptimizations::maxThreads
 * @brief Maximum number of concurrent threads allowed at a given time.
 */


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

    uint_t                     numberBasisFunctions_ = 100;
    PriorMeanFunction          priorMean_ = PriorMeanFunction::CONSTANT_PREPROCESS;
    CovarianceType             covarianceType_ = CovarianceType::VARIABLE_COVARIANCE;
    OutputUncertaintyType      outputUncertaintyType_ = OutputUncertaintyType::INPUT_DEPENDENT;
    WeightingScheme            weightingScheme_ = WeightingScheme::BALANCED;
    NormalizationScheme        normalizationScheme_ = NormalizationScheme::WHITEN;
    TrainValidationSplitMethod trainValidSplitMethod_ = TrainValidationSplitMethod::RANDOM;
    OptimizerMethod            optimizerMethod_ = OptimizerMethod::GPZ_LBFGS;
    double                     balancedWeightingBinSize_ = 0.1;
    double                     trainValidRatio_ = 0.5;
    uint_t                     optimizationMaxIterations_ = 200;
    double                     optimizationTolerance_ = 1e-9;
    double                     optimizationGradientTolerance_ = 1e-5;
    bool                       verbose_ = false;
    bool                       profileTraining_ = false;
    GPzOptimizations           optimizations_;

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

    GPzHyperParameters parameters_, derivatives_;

    // ====================
    // Projection variables
    // ====================

    Vec1d  featureMean_;       // GPz MatLab: muX
    Vec1d  featureSigma_;      // GPz MatLab: sdX
    double outputMean_ = 0.0;  // GPz MatLab: muY

    struct MissingCacheElement {
        // Base data

        int                id = 0;
        uint_t             countMissing = 0;       // u, o = d - u
        std::vector<bool>  missing;                // [d]
        std::vector<int>   indexMissing;           // [d]
        std::vector<int>   indexObserved;          // [d]

        // For initialization

        Mat2d              predictor;              // [u,o]

        // For training

        std::vector<Mat2d> covariancesObserved;    // [o,o], GPz MatLab: Sigma(o,o)
        std::vector<Mat2d> covariancesMissing;     // [u,u], GPz MatLab: Sigma(u,u)
        std::vector<Mat2d> invCovariancesObserved; // [o,o], GPz MatLab: inv(Sigma(o,o))
        Vec1d              covariancesObservedLogDeterminant; // GPz MatLab: lnz
        std::vector<Mat2d> gUO;                    // [u,o], GPz MatLab: GuuGuo
        std::vector<Mat2d> dgO;                    // [:,o], GPz MatLab: dGo/diSoo

        // For prediction

        std::vector<Mat2d> Psi_hat;                // [u,u], GPz MatLab: Psi_hat
        std::vector<Mat2d> R;                      // [o,u], GPz MatLab: R
        std::vector<Mat2d> T;                      // [:,o], GPz MatLab: T
        Mat2d              Nij;                    // [m,m], GPz MatLab: Nij
        std::vector<std::vector<Mat1d>> Nu;        // [m,m,m], GPz MatLab: Nu
    };

    mutable std::vector<MissingCacheElement> missingCache_;
    mutable MissingCacheElement* noMissingCache_ = nullptr;

    Vec1d  featurePCAMean_;
    Mat2d  featurePCASigma_;
    Mat2d  featurePCABasisVectors_;

    Mat1d  decorrelationCoefficients_;

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
    double sumWeightTrain_ = 0.0;
    Vec1i missingTrain_;
    Mat2d inputValid_;      // GPzMatLab: X[validation,:]
    Mat2d inputErrorValid_; // GPzMatLab: Psi[validation,:]
    Vec1d outputValid_;     // GPzMatLab: Y[validation,:]
    Vec1d weightValid_;     // GPzMatLab: omega[validation,:]
    double sumWeightValid_ = 0.0;
    Vec1i missingValid_;

    double logLikelihood_ = 0.0;
    double logLikelihoodValid_ = 0.0;

    // =============================
    // Minimization cached variables
    // =============================

    Mat2d trainBasisFunctions_; // GPzMatLab: PHI
    Mat1d trainOutputLogError_; // GPzMatLab: beta
    Mat2d validBasisFunctions_; // GPzMatLab: PHI
    Mat1d validOutputLogError_; // GPzMatLab: beta
    Mat1d modelWeights_;        // GPzMatLab: w
    Mat2d modelInvCovariance_;  // GPzMatLab: iSigma_w
    Mat1d modelInputPrior_;     // GPzMatLab: prior

    // ===========================
    // Prediction cached variables
    // ===========================

    mutable Mat2d lnZ; // GPz MatLab: lnZij

    // ====================================
    // Internal functions: hyper-parameters
    // ====================================

    Vec1d makeParameterArray_(const GPzHyperParameters& inputParams) const;

    void loadParametersArray_(const Vec1d& inputParams, GPzHyperParameters& outputParams) const;

    void resizeHyperParameters_(GPzHyperParameters& params) const;

    // ==================================
    // Internal functions: initialization
    // ==================================

    void updateNumberParameters_();

    void resizeArrays_();

    void reset_();

    void applyInputNormalization_(Mat2d& input, Mat2d& inputError) const;

    void applyOutputNormalization_(const Mat2d& input, const Vec1i& missing, Vec1d& output) const;

    void restoreOutputNormalization_(const Mat2d& input, const Vec1i& missing, GPzOutput& output) const;

    void computeWhitening_(const Mat2d& input);

    void computeLinearDecorrelation_(const Mat2d& input, const Mat2d& inputError,
        const Vec1d& output, const Vec1d& weight);

    void normalizeTrainingInputs_(Mat2d& input, Mat2d& inputError, const Vec1i& missing,
        Vec1d& output, const Vec1d& weight);

    void eraseInvalidTrainData_(Mat2d& input, Mat2d& inputError, Vec1d& output) const;

    void splitTrainValid_(const Mat2d& input, const Mat2d& inputError,
        const Vec1d& output, const Vec1d& weight);

    Vec1d computeWeights_(const Vec1d& output) const;

    void initializeInputs_(Mat2d input, Mat2d inputError, Vec1d output);

    void computeTrainingPCA_();

    void buildLinearPredictorCache_();

    void initializeBasisFunctions_();

    void initializeBasisFunctionRelevances_();

    void buildMissingCache_(const Mat2d& input) const;

    const MissingCacheElement* findMissingCacheElement_(int id) const;

    const MissingCacheElement& getMissingCacheElement_(int id) const;

    Vec1i getBestMissingID_(const Mat2d& input) const;

    void fetchMatrixElements_(Mat2d& out, const Mat2d& in, const MissingCacheElement& element,
        char first, char second) const;

    void fetchVectorElements_(Mat1d& out, const Mat1d& in, const MissingCacheElement& element,
        char first) const;

    void addMatrixElements_(Mat2d& out, const Mat2d& in, const MissingCacheElement& element,
        char first, char second) const;

    Mat2d initializeCovariancesFillLinear_(Mat2d input, const Vec1i& missing) const;

    Vec1d initializeCovariancesMakeGamma_(const Mat2d& input, const Vec1i& missing) const;

    void initializeCovariances_();

    void initializeErrors_();

    void initializeFit_();

    bool checkInputDimensions_(const Mat2d& input) const;

    bool checkErrorDimensions_(const Mat2d& input, const Mat2d& inputError) const;

    // =======================
    // Internal functions: fit
    // =======================

    Mat1d evaluateBasisFunctionsGeneral_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const;

    Mat1d evaluateBasisFunctionsDiag_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const;

    Mat1d evaluateBasisFunctions_(const Mat1d& input, const Mat1d& inputError, const MissingCacheElement& element) const;

    void updateBasisFunctions_(Mat2d& funcs, const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const;

    Mat2d evaluateBasisFunctions_(const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const;

    double evaluateOutputLogError_(const Mat1d& basisFunctions) const;

    Mat1d evaluateOutputLogErrors_(const Mat2d& basisFunctions) const;

    enum class MissingCacheUpdate {
        TRAIN, PREDICT
    };

    void updateMissingCache_(MissingCacheUpdate what) const;

    void updateTrainBasisFunctions_();

    void updateValidBasisFunctions_();

    void updateTrainOutputErrors_();

    void updateValidOutputErrors_();

    void updateTrainModel_(Minimize::FunctionOutput requested);

    void updateLikelihoodValid_();

    void computeInputPriors_();

    // ==============================
    // Internal functions: prediction
    // ==============================

    void predictFull_(const Mat1d& input, const MissingCacheElement& element, double& value,
        double& varianceTrainDensity, double& varianceTrainNoise) const;

    void predictNoisy_(const Mat1d& input, const Mat1d& inputError,
        const MissingCacheElement& element, double& value,
        double& varianceTrainDensity, double& varianceTrainNoise, double& varianceInputNoise) const;

    void predictMissingNoisy_(const Mat1d& input, const Mat1d& inputError,
        const MissingCacheElement& element, double& value,
        double& varianceTrainDensity, double& varianceTrainNoise, double& varianceInputNoise) const;

    GPzOutput predict_(const Mat2d& input, const Mat2d& inputError, const Vec1i& missing) const;

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
    GPz(const GPz&) = delete;

    /**
     * @brief Move constructor
     */
    GPz(GPz&&) = delete;

    /**
     * @brief Copy assignment operator
     */
    GPz& operator=(const GPz&) = delete;

    /**
     * @brief Move assignment operator
     */
    GPz& operator=(GPz&&) = delete;

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

    void setTrainValidationSplitMethod(TrainValidationSplitMethod method);

    TrainValidationSplitMethod getTrainValidationSplitMethod() const;

    void setTrainValidationRatio(double ratio);

    double getTrainValidationRatio() const;

    void setTrainValidationSplitSeed(uint_t seed);

    uint_t getTrainValidationSplitSeed() const;

    void setInitialPositionSeed(uint_t seed);

    uint_t getInitialPositionSeed() const;

    void setCovarianceType(CovarianceType type);

    CovarianceType getCovarianceType() const;

    void setOutputUncertaintyType(OutputUncertaintyType type);

    OutputUncertaintyType getOutputUncertaintyType() const;

    void setOptimizerMethod(OptimizerMethod method);

    OptimizerMethod getOptimizerMethod() const;

    void setOptimizationMaxIterations(uint_t maxIterations);

    uint_t getOptimizationMaxIterations() const;

    void setOptimizationTolerance(double tolerance);

    double getOptimizationTolerance() const;

    void setOptimizationGradientTolerance(double tolerance);

    double getOptimizationGradientTolerance() const;

    void setVerboseMode(bool verbose);

    bool getVerboseMode() const;

    void setOptimizationFlags(GPzOptimizations optimizations);

    GPzOptimizations getOptimizationFlags() const;

    void setProfileTraining(bool profile);

    // =====================
    // Fit/training function
    // =====================

    void fit(Mat2d input, Mat2d inputError, Vec1d output);

    // ========================
    // Fit model loading/saving
    // ========================

    void loadModel(const GPzModel& model);

    GPzModel getModel() const;

    // ===================
    // Prediction function
    // ===================

    GPzOutput predict(Mat2d input, Mat2d inputError) const;

};  // End of GPz class

}  // namespace PHZ_GPz


#endif
