/**
 * @file tests/src/GPz_test.cpp
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

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#if BOOST_VERSION / 100 % 1000 < 59
#define NO_BOOST_TEST
#endif
#if BOOST_VERSION / 100 % 1000 < 62
#define NO_BOOST_CONTEXT
#endif

#include "PHZ_GPz/GPz.h"
#include "PHZ_GPz/EigenWrapper.h"
#include "PHZ_GPz/STLWrapper.h"

#include <random>
#include <iostream>
#include <fstream>

using namespace PHZ_GPz;

class ModelChecker;

// Class to generate data sets with a known model
class Generator {
    friend class ModelChecker;

    std::mt19937 seed = std::mt19937(42);

    void makeValue(const Vec1d& input, double& output, double& uncertainty) {
        const uint_t d = numFeatures();
        const uint_t m = numBasisFunctions();

        output = 0.0;
        uncertainty = errorMin;
        for (uint_t j = 0; j < m; ++j) {
            double value = 0.0;
            for (uint_t k = 0; k < d; ++k) {
                value += square((input[k] - centerPos(j,k))/width[j]);
            }

            double bf = exp(-0.5*value);
            output += bf*ampMax[j];
            uncertainty += bf*errorMax[j];
        }

        uncertainty = sqrt(exp(uncertainty));
    }

public:

    Vec1d width;
    double errorMin;
    Vec1d errorMax;
    Vec1d ampMax;
    Vec2d centerPos;

    Vec2d testInput;
    Vec1d testTrueValue;
    Vec1d testTrueUncertainty;

    explicit Generator(uint_t d, uint_t m) :
        width(m), errorMax(m), ampMax(m), centerPos(m,d) {}

    uint_t numBasisFunctions() const {
        return width.size();
    }

    uint_t numFeatures() const {
        return centerPos.cols();
    }

    void makeTraining(uint_t n, Mat2d& input, Vec1d& output) {
        std::normal_distribution<double>       gauss(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        const uint_t d = numFeatures();

        input.resize(n,d);
        output.resize(n);

        for (uint_t i = 0; i < n; ++i) {
            for (uint_t k = 0; k < d; ++k) {
                input(i,k) = uniform(seed);
            }

            double out, unc;
            makeValue(input.row(i), out, unc);

            output[i] = gauss(seed)*unc + out;
        }
    }

    void makeTesting(uint_t n) {
        double minVal  = 0.0;
        double maxVal  = 1.0;
        double minTest = minVal - 0.1*(maxVal - minVal);
        double maxTest = maxVal + 0.1*(maxVal - minVal);

        const uint_t d = numFeatures();

        uint_t ndim = std::max(5.0, round(pow(n, 1.0/d)));
        n = 1;
        for (uint_t k = 0; k < d; ++k) {
            n *= ndim;
        }

        testInput.resize(n,d);
        testTrueValue.resize(n);
        testTrueUncertainty.resize(n);
        for (uint_t i = 0; i < n; ++i) {
            uint_t index = i;
            for (uint_t k = 0; k < d; ++k) {
                double a = (index % ndim)/float(ndim - 1);
                index = index / ndim;
                testInput(i,k) = minTest + a*(maxTest - minTest);
            }

            makeValue(testInput.row(i), testTrueValue[i], testTrueUncertainty[i]);
        }
    }
};

// Function to dump a training set on disk
void writeTraining(const std::string& filename, const Mat2d& input, const Vec1d& output) {
    const uint_t n = input.rows();
    const uint_t d = input.cols();

    std::ofstream out(filename);
    for (uint_t i = 0; i < n; ++i) {
        for (uint_t j = 0; j < d; ++j) {
            out << input(i,j) << " ";
        }
        out << output[i] << std::endl;
    }
}

// Function to dump a testing set on disk
void writeTesting(const std::string& filename, const Mat2d& input, const Vec1d& trueValue,
    const Vec1d& trueUncertainty, const Vec1d& observedValue, const Vec1d& observedUncertainty) {

    const uint_t n = input.rows();
    const uint_t d = input.cols();

    std::ofstream out(filename);
    for (uint_t i = 0; i < n; ++i) {
        for (uint_t j = 0; j < d; ++j) {
            out << input(i,j) << " ";
        }
        out << trueValue[i] << " " << trueUncertainty[i] << " "
            << observedValue[i] << " " << observedUncertainty[i] << std::endl;
    }
}

void writeTesting(const std::string& filename, const Mat2d& input, const Vec1d& trueValue,
    const Vec1d& trueUncertainty, const PHZ_GPz::GPzOutput& output) {

    writeTesting(filename, input, trueValue, trueUncertainty, output.value, output.uncertainty);
}

// Type traits required for StatisticAccumulator
template<typename Type>
struct TypeTraits {
    static void setZero(Type& t) {
        t = 0.0;
    }
};

template<>
struct TypeTraits<Vec1d> {
    static void setZero(Vec1d& t) {
        t.fill(0.0);
    }
};

template<>
struct TypeTraits<Vec2d> {
    static void setZero(Vec2d& t) {
        t.fill(0.0);
    }
};

// Class to accumulate statistics on a value over multiple tries
template<typename Type>
class StatisticAccumulator {
    Type value;
    Type valueSquared;
    uint_t count = 0;
    bool first = true;

    using Traits = TypeTraits<Type>;

public:

    void accumulate(const Type& v) {
        if (first) {
            value = v;
            Traits::setZero(value);
            valueSquared = v;
            Traits::setZero(valueSquared);
            first = false;
        }

        value += v;
        valueSquared += v*v;
        ++count;
    }

    Type mean() const {
        return value/count;
    }

    Type stddev() const {
        Type m = mean();
        return sqrt(valueSquared/count - m*m);
    }

    Type error() const {
        return stddev()/sqrt(count);
    }

    Type relativeError() const {
        return error()/mean();
    }
};

// Class to check the outcome of repeated tests in a given setup
class ModelChecker {
    const Generator& generator;

    StatisticAccumulator<Vec1d>  widthStacked;
    StatisticAccumulator<double> errorMinStacked;
    StatisticAccumulator<Vec1d>  errorMaxStacked;
    StatisticAccumulator<Vec1d>  ampMaxStacked;
    StatisticAccumulator<Vec2d>  centerPosStacked;
    StatisticAccumulator<Vec1d>  predValue;
    StatisticAccumulator<Vec1d>  predUncertainty;

    void setContext(const std::string& str) {
        #ifndef NO_BOOST_CONTEXT
        BOOST_TEST_INFO("with setup: " << setup);
        BOOST_TEST_INFO("at: " << str);
        #endif
    }

    void fillContext(std::ostringstream& obj) {}

    template<typename T, typename ... Args>
    void fillContext(std::ostringstream& obj, const T& t, const Args& ... args) {
        obj << t;
        fillContext(obj, args...);
    }

    template<typename ... Args>
    void setContext(const Args& ... args) {
        std::ostringstream obj;
        fillContext(obj, args...);
        setContext(obj.str());
    }

    void checkThreshold(double observed, double threshold) {
        #ifdef NO_BOOST_TEST
        BOOST_CHECK(observed < threshold);
        #else
        BOOST_TEST(observed < threshold);
        #endif
    }

    void checkDifference(double observed, double expected, double error, double fudge) {
        #ifdef NO_BOOST_TEST
        BOOST_CHECK(std::abs(observed - expected) < fudge*error);
        #else
        BOOST_TEST(observed == expected, tt::tolerance(fudge*error/expected));
        #endif
    }

    bool isBadDifference(double observed, double expected, double error, double fudge) {
        return std::abs(observed - expected) > fudge*error;
    }

public:

    std::string setup;

    explicit ModelChecker(const Generator& g) : generator(g) {}

    void accumulateModel(const GPzModel& model) {
        const uint_t m = generator.numBasisFunctions();
        const uint_t d = generator.numFeatures();

        Vec1d ampMaxModel = model.modelWeights;

        Vec2d centerPosModel = model.parameters.basisFunctionPositions;
        for (uint_t j = 0; j < m; ++j)
        for (uint_t k = 0; k < d; ++k) {
            centerPosModel(j,k) = centerPosModel(j,k)*model.featureSigma[k] + model.featureMean[k];
        }

        Vec1d widthModel(m);
        for (uint_t j = 0; j < m; ++j) {
            widthModel[j] = 0.0;
            for (uint_t k = 0; k < d; ++k) {
                widthModel[j] += (1.0/model.parameters.basisFunctionCovariances[j](k,k))
                                 *model.featureSigma[k];
            }

            widthModel[j] /= d;
        }

        double errorMinModel = model.parameters.logUncertaintyConstant;
        Vec1d errorMaxModel = model.parameters.uncertaintyBasisWeights;

        ampMaxStacked.accumulate(ampMaxModel);
        centerPosStacked.accumulate(centerPosModel);
        widthStacked.accumulate(widthModel);
        errorMinStacked.accumulate(errorMinModel);
        errorMaxStacked.accumulate(errorMaxModel);
    }

    void accumulatePrediction(const GPzOutput& result) {
        predValue.accumulate(result.value);
        predUncertainty.accumulate(result.uncertainty);
    }

    void checkErrors(double threshold) {
        const uint_t m = generator.numBasisFunctions();
        const uint_t d = generator.numFeatures();

        setContext("errorMin relative error");
        checkThreshold(errorMinStacked.relativeError(), threshold);

        for (uint_t j = 0; j < m; ++j) {
            setContext("ampMax relative error for BF ", j);
            checkThreshold(ampMaxStacked.relativeError()[j], threshold);
            setContext("width relative error for BF ", j);
            checkThreshold(widthStacked.relativeError()[j], threshold);
            setContext("errorMax relative error for BF ", j);
            checkThreshold(errorMaxStacked.relativeError()[j], threshold);
            for (uint_t k = 0; k < d; ++k) {
                setContext("centerPos relative error for BF ", j, " and feature ", k);
                checkThreshold(centerPosStacked.relativeError()(j,k), threshold);
            }
        }
    }

    void checkValues(double fudge) {
        const uint_t m = generator.numBasisFunctions();
        const uint_t d = generator.numFeatures();

        setContext("errorMin value");
        checkDifference(errorMinStacked.mean(), generator.errorMin, errorMinStacked.error(), fudge);

        for (uint_t j = 0; j < m; ++j) {
            setContext("ampMax value for BF ", j);
            checkDifference(ampMaxStacked.mean()[j], generator.ampMax[j], ampMaxStacked.error()[j], fudge);
            setContext("width value for BF ", j);
            checkDifference(widthStacked.mean()[j], generator.width[j], widthStacked.error()[j], fudge);
            setContext("errorMax value for BF ", j);
            checkDifference(errorMaxStacked.mean()[j], generator.errorMax[j], errorMaxStacked.error()[j], fudge);
            for (uint_t k = 0; k < d; ++k) {
                setContext("centerPos value for BF ", j, " and feature ", k);
                checkDifference(centerPosStacked.mean()(j,k), generator.centerPos(j,k), centerPosStacked.error()(j,k), fudge);
            }
        }
    }

    void checkPrediction(double fudge) {
        const uint_t n = generator.testTrueValue.size();

        Vec1d value = predValue.mean();
        Vec1d valueError = predValue.error();
        Vec1d uncertainty = predUncertainty.mean();
        Vec1d uncertaintyError = predUncertainty.error();

        writeTesting("testing2.txt", generator.testInput, generator.testTrueValue,
            generator.testTrueUncertainty, value, valueError);

        setContext("predicted value error rate");
        uint_t numBad = 0;
        for (uint_t i = 0; i < n; ++i) {
            if (valueError[i])
            if (isBadDifference(value[i], generator.testTrueValue[i], valueError[i], fudge)) {
                ++numBad;
            }
        }

        checkThreshold(numBad/float(n), 0.05);

        setContext("predicted uncertainty error rate");
        numBad = 0;
        for (uint_t i = 0; i < n; ++i) {
            if (isBadDifference(uncertainty[i], generator.testTrueUncertainty[i], uncertaintyError[i], fudge)) {
                ++numBad;
            }
        }

        checkThreshold(numBad/float(n), 0.05);
    }
};

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE (GPz_test_train)

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE( train_1D ) {

    const bool debugTest = false;

    for (uint_t setup = 0; setup <= 10; ++setup) {
        Mat2d input;
        Vec1d output;

        Generator generator(1,1);

        generator.errorMin = 0.3;
        generator.width << 0.1;
        generator.errorMax << 1.0;
        generator.ampMax << 3.0;
        generator.centerPos << 0.5;

        generator.makeTesting(200);

        ModelChecker checker(generator);

        uint_t nstack = 20;

        GPz gpz;
        gpz.setVerboseMode(false);
        gpz.setNumberOfBasisFunctions(1);
        gpz.setPriorMeanFunction(PriorMeanFunction::ZERO);

        GPzOptimizations opts;

        bool setErrors = false;
        bool setWeights = false;
        bool setHints = false;

        switch (setup) {
            case 0: {
                // Default
                checker.setup = "default (GPVC, no fuzzing, no error, no weight)";
                break;
            }
            case 1: {
                // With fuzzing
                checker.setup = "fuzzing";
                gpz.setFuzzInitialValues(true);
                break;
            }
            case 2: {
                // With optimizations disabled
                checker.setup = "no optimization";
                opts.specializeForSingleFeature = false;
                opts.specializeForDiagCovariance = false;
                break;
            }
            case 3: {
                // With different covariance prescription
                checker.setup = "GPGL";
                gpz.setCovarianceType(CovarianceType::GLOBAL_LENGTH);
                break;
            }
            case 4: {
                // With different covariance prescription
                checker.setup = "GPGD";
                gpz.setCovarianceType(CovarianceType::GLOBAL_DIAGONAL);
                break;
            }
            case 5: {
                // With different covariance prescription
                checker.setup = "GPGV";
                gpz.setCovarianceType(CovarianceType::GLOBAL_COVARIANCE);
                break;
            }
            case 6: {
                // With different covariance prescription
                checker.setup = "GPVL";
                gpz.setCovarianceType(CovarianceType::VARIABLE_LENGTH);
                break;
            }
            case 7: {
                // With different covariance prescription
                checker.setup = "GPVD";
                gpz.setCovarianceType(CovarianceType::VARIABLE_DIAGONAL);
                break;
            }
            case 8: {
                // With zero errors (see below)
                checker.setup = "with zero errors";
                setErrors = true;
                break;
            }
            case 9: {
                // With unity & uniform weights (see below)
                checker.setup = "with unity weights";
                setWeights = true;
                break;
            }
            case 10: {
                // With hints (see below)
                checker.setup = "with hints";
                setHints = true;
                break;
            }
        }

        gpz.setOptimizationFlags(opts);

        for (uint_t s = 0; s < nstack; ++s) {
            // Prepare training data
            generator.makeTraining(10000, input, output);

            if (s == 0 && debugTest) {
                writeTraining("train.txt", input, output);
            }

            // Do training
            gpz.setInitialPositionSeed(66 + s);
            gpz.setFuzzingSeed(77 + s);

            Mat2d errors;
            Vec1d weights;
            GPzHints hints;
            if (setup == 8) {
                errors.setZero(input.rows(), 1);
            }
            if (setup == 9) {
                weights.resize(input.rows());
                weights.fill(1.0);
            }
            if (setup == 10) {
                hints.basisFunctionPositions.resize(1,1);
                hints.basisFunctionPositions(0,0) = generator.centerPos(0,0)+0.1;
            }

            gpz.fit(input, errors, output, weights, hints);

            // Fetch model parameters
            checker.accumulateModel(gpz.getModel());

            // Do testing
            auto result = gpz.predict(generator.testInput, Mat2d{});

            if (s == 0 && debugTest) {
                writeTesting("test.txt", generator.testInput,
                    generator.testTrueValue, generator.testTrueUncertainty, result);
            }

            // Fetch predicted values
            checker.accumulatePrediction(result);
        }

        // Test that all attempts converged to a similar model (at better than 2%)
        checker.checkErrors(0.02);

        // Test that all attempts converged to the right model within noise
        checker.checkValues(10.0);

        // Test that all the predictions converged to the right values within noise
        checker.checkPrediction(4.0);
    }

}

BOOST_AUTO_TEST_CASE( train_2D ) {

    const bool debugTest = true;

    for (uint_t setup = 0; setup <= 6; ++setup) {
        Mat2d input;
        Vec1d output;

        Generator generator(2,2);

        generator.errorMin = 0.3;
        generator.width << 0.1, 0.1;
        generator.errorMax << 1.0, 0.2;
        generator.ampMax << 3.0, 5.0;
        generator.centerPos << 0.66, 0.66, 0.1, 0.1;

        generator.makeTesting(200*200);

        ModelChecker checker(generator);

        uint_t nstack = 20;

        GPz gpz;
        gpz.setVerboseMode(false);
        gpz.setNumberOfBasisFunctions(100);
        gpz.setPriorMeanFunction(PriorMeanFunction::ZERO);

        GPzOptimizations opts;
        opts.enableMultithreading = true;
        opts.maxThreads = 4;

        bool setErrors = false;
        bool setWeights = false;

        switch (setup) {
            case 0: {
                // Default
                checker.setup = "default (GPVC, no fuzzing, no error, no weight)";
                break;
            }
            case 1: {
                // With fuzzing
                checker.setup = "fuzzing";
                gpz.setFuzzInitialValues(true);
                break;
            }
            case 2: {
                // With optimizations disabled
                checker.setup = "no optimization";
                opts.specializeForSingleFeature = false;
                opts.specializeForDiagCovariance = false;
                // opts.enableMultithreading = false;
                break;
            }
            case 3: {
                // With different covariance prescription
                checker.setup = "GPVL";
                gpz.setCovarianceType(CovarianceType::VARIABLE_LENGTH);
                break;
            }
            case 4: {
                // With different covariance prescription
                checker.setup = "GPVD";
                gpz.setCovarianceType(CovarianceType::VARIABLE_DIAGONAL);
                break;
            }
            case 5: {
                // With zero errors (see below)
                checker.setup = "with zero errors";
                setErrors = true;
                break;
            }
            case 6: {
                // With unity & uniform weights (see below)
                checker.setup = "with unity weights";
                setWeights = true;
                break;
            }
        }

        gpz.setOptimizationFlags(opts);

        for (uint_t s = 0; s < nstack; ++s) {
            // Prepare training data
            generator.makeTraining(10000, input, output);

            if (s == 0 && debugTest) {
                writeTraining("train.txt", input, output);
            }

            // Do training
            gpz.setInitialPositionSeed(66 + s);
            gpz.setFuzzingSeed(77 + s);

            Mat2d errors;
            Vec1d weights;
            if (setErrors) {
                errors.setZero(input.rows(), 2);
            }
            if (setWeights) {
                weights.resize(input.rows());
                weights.fill(1.0);
            }

            gpz.fit(input, errors, output, weights);

            // Do testing
            auto result = gpz.predict(generator.testInput, Mat2d{});

            if (s == 0 && debugTest) {
                writeTesting("test.txt", generator.testInput,
                    generator.testTrueValue, generator.testTrueUncertainty, result);
            }

            // Fetch predicted values
            checker.accumulatePrediction(result);
        }

        // Test that all the predictions converged to the right values within noise
        checker.checkPrediction(4.0);
    }

}

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE_END ()


