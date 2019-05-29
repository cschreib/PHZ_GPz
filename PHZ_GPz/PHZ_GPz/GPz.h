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
#include "PHZ_GPz/Minimize.h"
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
 * (one plus one per input feature), and allows the most faithful predictions outside of the training
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
 * Whitening is the process of subtracting the data mean and dividing by the data standard deviation,
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
 * @brief Use the custom L-BFGS algorithm as implemented in GPz MatLab (default). For this
 *        optimizer, the recommended optimization tolerance threshold is 1e-9.
 */

/**
 * @struct GPzOutput
 * @brief Store the output of a GPz run
 *
 * Note: uncertainty = sqrt(varianceTrainDensity + varianceTrainNoise + varianceInputNoise)
 *
 */
struct GPzOutput {
    Vec1d value;                ///< Predicted value
    Vec1d uncertainty;          ///< Predicted total uncertainty
    Vec1d varianceTrainDensity; ///< Predicted variance due to density of training data
    Vec1d varianceTrainNoise;   ///< Predicted variance due to training data noise
    Vec1d varianceInputNoise;   ///< Predicted variance due to input noise
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
    double                     balancedWeightingMaxWeight_ = 10.0;
    double                     trainValidRatio_ = 0.5;
    uint_t                     optimizationMaxIterations_ = 500;
    double                     optimizationTolerance_ = 1e-9;
    double                     optimizationGradientTolerance_ = 1e-5;
    bool                       predictVariance_ = true;
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

    void setNumberOfFeatures_(uint_t num);

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

    void eraseInvalidTrainData_(Mat2d& input, Mat2d& inputError, Vec1d& output, Vec1d& weight) const;

    void splitTrainValid_(const Mat2d& input, const Mat2d& inputError,
        const Vec1d& output, const Vec1d& weight);

    Vec1d computeWeights_(const Vec1d& output) const;

    void initializeInputs_(Mat2d input, Mat2d inputError, Vec1d output, Vec1d weight);

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

    /**
     * @name Training
     */
    /**@{*/

    // =====================
    // Fit/training function
    // =====================

    /**
     * @brief Train the GPz model
     *
     * \param input 2D array of feature values for each element of the training set
     *              (dimensions: N_element x N_feature)
     * \param inputError 2D array of feature uncertainties associated to the input feature values
     *                   (dimensions: N_element x N_feature; or empty if there are no uncertainties)
     * \param output 1D array of output values to train against (dimensions: N_element)
     * \param weight 1D array of weights to use in training (dimensions: N_element; or empty to let
     *               GPz determine the weights, see setWeightingScheme())
     *
     * Invalid values (infinite or not-a-number) in the inputs and/or in the input uncertainties
     * will be considered as missing features and will be treated accordingly. If some elements of
     * the training set have all their features missing, or if their output value and/or weight are
     * invalid, they will be discarded and will not participate to the training.
     *
     * The input errors should be given as "one sigma" uncertainties, not variances. Zero is an
     * acceptable value, and signifies that there is no uncertainty.
     *
     * It is not possible to set bounds on the output values. If your output space is bounded (for
     * example, if negative values are not permitted) you should transform your output values to
     * an un-bounded output space (for example, using a logarithm for strictly positive outputs).
     *
     * Once the training is complete, you may use the getModel() and predict() functions.
     */
    void fit(Mat2d input, Mat2d inputError, Vec1d output, Vec1d weight);

    /**@}*/

    /**
     * @name Accessing the GPz model
     */
    /**@{*/

    // ========================
    // Fit model loading/saving
    // ========================

    /**
     * @brief Load a pre-trained model
     *
     * The training stage can be time-consuming, but it usually only needs to be done once. After
     * the training is complete, the resulting model parameters can be accessed with getNodel() to
     * be saved somewhere, for example by writing them somewhere on the hard drive, and re-loaded
     * later for reuse with loadModel().
     *
     * You can also use this function to experiment with the GPz model and manually input model
     * parameters. The function will check that the parameter values are valid and will throw an
     * exception if anything is wrong; in this case the internal model of the class will be left
     * unchanged. Else, if a model previously existed in the class, it will be discarded.
     */
    void loadModel(const GPzModel& model);

    /**
     * @brief Get a copy of the current model
     *
     * See loadModel(). The returned model is only a copy of the model that is internally
     * used by the class; it is not a reference. If you make modifications to the output value of
     * this function, you have to load it back in with loadModel() for your changes to take effect.
     *
     * \pre A model must currently exist in the class, for example by calling fit() or loadModel().
     * If not, the function will throw an exception.
     */
    GPzModel getModel() const;

    /**@}*/

    /**
     * @name Predicting
     */
    /**@{*/

    // ===================
    // Prediction function
    // ===================

    /**
     * @brief Make a prediction or test using the current model
     *
     * \param input 2D array of feature values for each element of the testing set
     *              (dimensions: N_element x N_feature)
     * \param inputError 2D array of feature uncertainties associated to the input feature values
     *                   (dimensions: N_element x N_feature; or empty if there are no uncertainties)
     * \return A GPzOutput struct holding the results of the prediction, including predicted values
     *         and predicted uncertainties.
     *
     * Invalid values (infinite or not-a-number) in the inputs and/or in the input uncertainties
     * will be considered as missing features and will be treated accordingly. If some elements of
     * the testing set have all their features missing, GPz will still make a prediction.
     *
     * The input errors should be given as "one sigma" uncertainties, not variances. Zero is an
     * acceptable value, and signifies that there is no uncertainty.
     *
     * \pre A model must currently exist in the class, for example by calling fit() or loadModel().
     * If not, the function will throw an exception.
     */
    GPzOutput predict(Mat2d input, Mat2d inputError) const;

    /**@}*/

    // =============================
    // Configuration getters/setters
    // =============================

    /**
     * @name Getters/setters to configure training
     */
    /**@{*/

    /**
     * @brief Set the number of basis functions in the sparse GP
     *
     * The basis functions are D-dimensional Gaussian (for D features) of variable width (or
     * covariance) and amplitude, which are used to model the relation between the output values
     * and the input values in the training data set. As the number of basis function increases,
     * the model gains in richness and complexity; this increases the quality of the model at the
     * expense of computation time. In principle there should be no degradation of quality if too
     * many basis functions are used; GPz has automatic relevance determination (ARD) which will
     * weight down the basis functions which are not needed for modeling the data.
     *
     * Default value: 100
     */
    void setNumberOfBasisFunctions(uint_t num);

    /**
     * @brief Return the number of basis functions in the sparse GP
     *
     * See setNumberOfBasisFunctions().
     */
    uint_t getNumberOfBasisFunctions() const;

    /**
     * @brief Set the functional form of the prior mean function
     *
     * See PriorMeanFunction.
     *
     * Default value: PriorMeanFunction::CONSTANT_PREPROCESS.
     */
    void setPriorMeanFunction(PriorMeanFunction newFunction);

    /**
     * @brief Return the functional form of the prior mean function
     *
     * See setPriorMeanFunction().
     */
    PriorMeanFunction getPriorMeanFunction() const;

    /**
     * @brief Return the number of features that the current model has been trained for
     *
     * If no model has been trained yet, will return zero.
     */
    uint_t getNumberOfFeatures() const;

    /**
     * @brief Set the scheme used for weighting the training data
     *
     * See WeightingScheme.
     *
     * Default value: WeightingScheme::BALANCED.
     */
    void setWeightingScheme(WeightingScheme scheme);

    /**
     * @brief Return the scheme used for weighting the training data
     *
     * See setWeightingScheme().
     */
    WeightingScheme getWeightingScheme() const;

    /**
     * @brief Set the bin size for the balanced weighting scheme
     *
     * See WeightingScheme::BALANCED. In the "balanced" weighting scheme, a grid is created in
     * output space and the number of elements of the training set falling in each grid cell is
     * computed. The inverse of this number is then used as weighting for each element of the cell,
     * which effectively down-weights the regions of the parameter space that are over-represented.
     * This parameter controls the size of a grid cell, in the same units as used for the output
     * values.
     *
     * Default value: 0.1.
     */
    void setBalancedWeightingBinSize(double size);

    /**
     * @brief Return the bin size for the balanced weighting scheme
     *
     * See setBalancedWeightingBinSize().
     */
    double getBalancedWeightingBinSize() const;

    /**
     * @brief Set the maximum weight allowed in the balanced weighting scheme
     *
     * See WeightingScheme::BALANCED. In the "balanced" weighting scheme, elements of the training
     * set are weighted according to inverse density, to down-weights the regions of the parameter
     * space that are over-represented. In the most extreme cases, this can lead to differences in
     * weights of several orders of magnitude, putting a large weight on a few elements. To mitigate
     * this, the method imposes a maximum relative weight between the highest and lowest weighted
     * elements, which is controlled by this value.
     *
     * This behavior can be disabled (allowing any weight) by setting this value to not-a-number.
     *
     * Default value: 10.
     */
    void setBalancedWeightingMaxWeight(double weight);

    /**
     * @brief Return the maximum weight allowed in the balanced weighting scheme
     *
     * See setBalancedWeightingMaxWeight().
     */
    double getBalancedWeightingMaxWeight() const;

    /**
     * @brief Set the scheme used to pre-normalize the training data
     *
     * See NormalizationScheme.
     *
     * Default value: NormalizationScheme::WHITEN.
     */
    void setNormalizationScheme(NormalizationScheme scheme);

    /**
     * @brief Return the scheme used to pre-normalize the training data
     *
     * See setNormalizationScheme().
     */
    NormalizationScheme getNormalizationScheme() const;

    /**
     * @brief Set the method used to split training data into training and validation
     *
     * See TrainValidationSplitMethod, setTrainValidationRatio(), and setTrainValidationSplitSeed().
     *
     * Default value: TrainValidationSplitMethod::RANDOM.
     */
    void setTrainValidationSplitMethod(TrainValidationSplitMethod method);

    /**
     * @brief Return the method used to split training data into training and validation
     *
     * See setTrainValidationSplitMethod().
     */
    TrainValidationSplitMethod getTrainValidationSplitMethod() const;

    /**
     * @brief Set the fraction of training data to use for validation
     *
     * See TrainValidationSplitMethod and setTrainValidationSplitSeed().
     *
     * The training stage is performed by optimizing the likelihood of the model on the training
     * data. To avoid over-fitting, some of that training data is kept aside and does not enter
     * the training likelihood evaluation; this is the validation set. At each step of the
     * optimization routine, the likelihood of the validation set is computed separately, and the
     * optimization stops when this likelihood no longer improves.
     *
     * Default value: 0.5 (half).
     */
    void setTrainValidationRatio(double ratio);

    /**
     * @brief Return the fraction of training data to use for validation
     *
     * See setTrainValidationRatio().
     */
    double getTrainValidationRatio() const;

    /**
     * @brief Set the random seed to use for splitting the training data
     *
     * See TrainValidationSplitMethod and setTrainValidationRatio().
     *
     * If the training split method is set to TrainValidationSplitMethod::RANDOM, this value
     * is used to initialize the random number generator that decides whether an element of the
     * training set should be used for training or validation. If always initialized with the
     * same seed, the random number generator will always make the same decision, which is
     * desirable for reproducibility. Changing the seed even slightly will produce a completely
     * different result.
     *
     * Default value: 42.
     */
    void setTrainValidationSplitSeed(uint_t seed);

    /**
     * @brief Return the random seed to use for splitting the training data
     *
     * See setTrainValidationSplitSeed().
     */
    uint_t getTrainValidationSplitSeed() const;

    /**
     * @brief Set the random seed to use for initializing the basis function positions
     *
     * This value is used to initialize the random number generator that decides where to place
     * the basis functions in the initialization stage, before the optimization. If always
     * initialized with the same seed, the random number generator will always make the same
     * decision, which is desirable for reproducibility. Changing the seed even slightly will
     * produce a completely different result.
     *
     * Default value: 55.
     */
    void setInitialPositionSeed(uint_t seed);

    /**
     * @brief Return the random seed to use for initializing the basis function positions
     *
     * See setInitialPositionSeed().
     */
    uint_t getInitialPositionSeed() const;

    /**
     * @brief Set the basis function covariance type
     *
     * See CovarianceType.
     *
     * Default value: CovarianceType::VARIABLE_COVARIANCE.
     */
    void setCovarianceType(CovarianceType type);

    /**
     * @brief Return the basis function covariance type
     *
     * See setCovarianceType().
     */
    CovarianceType getCovarianceType() const;

    /**
     * @brief Set the output uncertainty parametrization type
     *
     * See OutputUncertaintyType.
     *
     * Default value: OutputUncertaintyType::INPUT_DEPENDENT.
     */
    void setOutputUncertaintyType(OutputUncertaintyType type);

    /**
     * @brief Return the output uncertainty parametrization type
     *
     * See setOutputUncertaintyType().
     */
    OutputUncertaintyType getOutputUncertaintyType() const;

    /**
     * @brief Set the optimizer implementation to use for training
     *
     * See OptimizerMethod.
     *
     * Default value: OptimizerMethod::GPZ_LBFGS.
     */
    void setOptimizerMethod(OptimizerMethod method);

    /**
     * @brief Return the optimizer implementation to use for training
     *
     * See setOptimizerMethod.
     */
    OptimizerMethod getOptimizerMethod() const;

    /**
     * @brief Set the maximum number of iteration of the optimization routine
     *
     * See setOptimizationTolerance() and setOptimizationGradientTolerance().
     *
     * The training is done with an iterative gradient search algorithm (BFGS, or LBFGS).
     * These optimization routines are designed to stop automatically when some convergence criteria
     * are reached (for example, the likelihood did not vary significantly compared to the previous
     * iteration, or the gradient is small enough). The number of iterations required for
     * convergence varies significantly between different setups, with the number of features, the
     * number of model parameters, but also the data itself. In most cases, decent solutions are
     * found after one or two hundred iterations, and the remaining iterations are used for fine
     * tuning (which may or may not be useful), until the convergence is reached formally.
     *
     * However, numerical instabilities can sometimes prevent the convergence criteria from ever
     * being reached. For this reason the optimization routines also set a hard limit on the
     * number of iterations, to avoid an infinite loop. If the optimization stops because the
     * maximum number of iteration has been reached, it is worth taking a good look at the resulting
     * model to make sure it is sensible.
     *
     * Default value: 500.
     */
    void setOptimizationMaxIterations(uint_t maxIterations);

    /**
     * @brief Return the maximum number of iteration of the optimization routine
     *
     * See setOptimizationMaxIterations().
     */
    uint_t getOptimizationMaxIterations() const;

    /**
     * @brief Set the optimization tolerance threshold use for convergence
     *
     * See setOptimizationMaxIterations() and setOptimizationGradientTolerance().
     *
     * The training is done with an iterative gradient search algorithm (BFGS, or LBFGS).
     * These optimization routines are designed to stop automatically when some convergence criteria
     * are reached (for example, the likelihood did not vary significantly compared to the previous
     * iteration, or the gradient is small enough). This threshold is used for one of these criteria
     * and its meaning depends on the adopted optimization method (see OptimizerMethod). Generally,
     * the lower the value, the more accurate the training will be, at the expense of increasing the
     * number of iteration required for convergence. However, a too low value may bring the
     * convergence criterion too close to numerical noise, which can make it difficult or impossible
     * to reach.
     *
     * For OptimizerMethod::GPZ_LBFGS (the default), this value is compared against the maximum
     * parameter step size in an iteration, and on the absolute value of the likelihood difference.
     * Convergence is reached if either is below this threshold. The recommended value is 1e-9.
     *
     * For OptimizerMethod::GSL_BFGS, the meaning of this value is undocumented, but a recommended
     * value is 0.1.
     *
     * Default value: 1e-9.
     */
    void setOptimizationTolerance(double tolerance);

    /**
     * @brief Return the optimization tolerance threshold use for convergence
     *
     * See setOptimizationTolerance().
     */
    double getOptimizationTolerance() const;

    /**
     * @brief Set the optimization tolerance threshold use for convergence
     *
     * See setOptimizationMaxIterations() and setOptimizationTolerance().
     *
     * The training is done with an iterative gradient search algorithm (BFGS, or LBFGS).
     * These optimization routines are designed to stop automatically when some convergence criteria
     * are reached (for example, the likelihood did not vary significantly compared to the previous
     * iteration, or the gradient is small enough). This threshold is used for one of these criteria
     * and its meaning depends on the adopted optimization method (see OptimizerMethod). Generally,
     * the lower the value, the more accurate the training will be, at the expense of increasing the
     * number of iteration required for convergence. However, a too low value may bring the
     * convergence criterion too close to numerical noise, which can make it difficult or impossible
     * to reach.
     *
     * For OptimizerMethod::GPZ_LBFGS (the default), this value is compared against the absolute
     * value of the gradient at each iteration. Convergence is reached if no parameter has its
     * gradient larger than this threshold. The recommended value is 1e-5.
     *
     * For OptimizerMethod::GSL_BFGS, this value is compared against the norm of the gradient at
     * each iteration. Convergence is reached if the norm is lower than this threshold. The
     * documentation does not provide a recommended value, but 1e-5 provides acceptable results.
     *
     * Default value: 1e-5.
     */
    void setOptimizationGradientTolerance(double tolerance);

    /**
     * @brief Return the optimization tolerance threshold use for convergence
     *
     * See setOptimizationGradientTolerance().
     */
    double getOptimizationGradientTolerance() const;

    /**
     * @brief Enable/disable predicting output variance in addition to point estimates
     *
     * Predicting variances is typically the most time-consuming part of the prediction stage.
     * If these variances are not required (i.e., when testing various setups to figure out
     * which provides the best point estimates), the computation can be disabled to save time.
     *
     * Default value: true.
     */
    void setPredictVariance(bool predictVariance);

    /**
     * @brief Check if predicting output variances in addition to point estimates
     *
     * See setPredictVariance().
     */
    bool getPredictVariance() const;

    /**
     * @brief Enable/disable printing the progress of the work to the standard output
     *
     * By default the program will not output anything in the standard output, so you will have
     * no feedback as to its progress. This is OK for un-supervised runs, but if you are running
     * GPz for the first time on a new data set, it can be useful to check what the code is doing,
     * and get an estimate of how long it will take to run. Enabling the "verbose" mode allows you
     * to do so, with a very minimal impact on performances.
     *
     * Default value: false (no output).
     */
    void setVerboseMode(bool verbose);

    /**
     * @brief Check if printing the progress of the work to the standard output
     *
     * See setVerboseMode().
     */
    bool getVerboseMode() const;

    /**
     * @brief Set optimization to use when evaluating the model and its derivatives
     *
     * See GPzOptimizations.
     *
     * Default value: all optimizations enabled.
     */
    void setOptimizationFlags(GPzOptimizations optimizations);

    /**
     * @brief Return the optimization to use when evaluating the model and its derivatives
     *
     * See setOptimizationFlags().
     */
    GPzOptimizations getOptimizationFlags() const;

    /**
     * @brief Enable/disable execution time profiling in training (for debug only)
     *
     * Default value: false (disabled).
     */
    void setProfileTraining(bool profile);

    /**@}*/

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
     * @brief Copy constructor (deleted)
     */
    GPz(const GPz&) = delete;

    /**
     * @brief Move constructor (deleted)
     */
    GPz(GPz&&) = delete;

    /**
     * @brief Copy assignment operator (deleted)
     */
    GPz& operator=(const GPz&) = delete;

    /**
     * @brief Move assignment operator (deleted)
     */
    GPz& operator=(GPz&&) = delete;

};  // End of GPz class

}  // namespace PHZ_GPz


#endif
