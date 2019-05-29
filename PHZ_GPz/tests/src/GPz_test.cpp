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

#include "PHZ_GPz/GPz.h"
#include "PHZ_GPz/EigenWrapper.h"
#include "PHZ_GPz/STLWrapper.h"

#include <random>
#include <iostream>
#include <fstream>

using namespace PHZ_GPz;

class Generator1D {
    std::mt19937 seed = std::mt19937(42);

public:

    double width     = 0.1;
    double errorMin  = 0.3;
    double errorMax  = 1.0;
    double ampMax    = 3.0;
    double centerPos = 0.5;

    void makeTraining(uint_t n, Mat2d& input, Vec1d& output) {
        std::normal_distribution<double>       gauss(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        input.resize(n,1);
        output.resize(n);

        for (uint_t i = 0; i < n; ++i) {
            input(i,0) = uniform(seed);
            double bf = exp(-0.5*square((input(i,0) - centerPos)/width));
            output[i] = gauss(seed)*sqrt(exp(errorMin + errorMax*bf)) + ampMax*bf;
        }
    }

    void makeTesting(uint_t n, Mat2d& input, Vec1d& trueValue, Vec1d& trueUncertainty) {
        double minVal  = 0.0;
        double maxVal  = 1.0;
        double minTest = minVal - 0.1*(maxVal - minVal);
        double maxTest = maxVal + 0.1*(maxVal - minVal);

        input.resize(n,1);
        trueValue.resize(n);
        trueUncertainty.resize(n);
        for (uint_t i = 0; i < n; ++i) {
            input(i,0) = minTest + i*(maxTest - minTest)/n;

            double bf = exp(-0.5*square((input(i,0) - centerPos)/width));
            trueValue[i] = ampMax*bf;
            trueUncertainty[i] = sqrt(exp(errorMin + errorMax*bf));
        }
    }
};

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

void writeTesting(const std::string& filename, const Mat2d& input, const Vec1d& trueValue,
    const Vec1d& trueUncertainty, const PHZ_GPz::GPzOutput& output) {

    const uint_t n = input.rows();
    const uint_t d = input.cols();

    std::ofstream out(filename);
    for (uint_t i = 0; i < n; ++i) {
        for (uint_t j = 0; j < d; ++j) {
            out << input(i,j) << " ";
        }
        out << trueValue[i] << " " << trueUncertainty[i] << " "
            << output.value[i] << " " << output.uncertainty[i] << std::endl;
    }
}

template<typename Type>
struct TypeTraits {
    static uint_t size(const Type&) {
        return 0;
    }

    static void resize(Type&, uint_t) {}

    static void setZero(Type& t) {
        t = 0.0;
    }
};

template<>
struct TypeTraits<Vec1d> {
    static uint_t size(const Vec1d& t) {
        return t.size();
    }

    static void resize(Vec1d& t, uint_t n) {
        t.resize(n);
    }

    static void setZero(Vec1d& t) {
        t.fill(0.0);
    }
};

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
            Traits::resize(value, Traits::size(v));
            Traits::setZero(value);
            Traits::resize(valueSquared, Traits::size(v));
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

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE (GPz_test_train)

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE( train_1D ) {

    const bool debugTest = false;

    for (uint_t setup = 0; setup <= 9; ++setup) {
        Mat2d input;
        Vec1d output;
        Vec1d trueValue, trueUncertainty;

        Generator1D generator;

        StatisticAccumulator<double> widthStacked;
        StatisticAccumulator<double> errorMinStacked;
        StatisticAccumulator<double> errorMaxStacked;
        StatisticAccumulator<double> ampMaxStacked;
        StatisticAccumulator<double> centerPosStacked;
        StatisticAccumulator<Vec1d>  predValue;
        StatisticAccumulator<Vec1d>  predUncertainty;

        uint_t nstack = 20;

        GPz gpz;
        gpz.setVerboseMode(false);
        gpz.setNumberOfBasisFunctions(1);
        gpz.setPriorMeanFunction(PriorMeanFunction::ZERO);

        GPzOptimizations opts;

        switch (setup) {
            case 0: {
                // Default
                break;
            }
            case 1: {
                // With fuzzing
                gpz.setFuzzInitialValues(true);
                break;
            }
            case 2: {
                // With optimizations disabled
                opts.specializeForSingleFeature = false;
                opts.specializeForDiagCovariance = false;
                break;
            }
            case 3: {
                // With different covariance prescription
                gpz.setCovarianceType(CovarianceType::GLOBAL_LENGTH);
                break;
            }
            case 4: {
                // With different covariance prescription
                gpz.setCovarianceType(CovarianceType::GLOBAL_DIAGONAL);
                break;
            }
            case 5: {
                // With different covariance prescription
                gpz.setCovarianceType(CovarianceType::GLOBAL_COVARIANCE);
                break;
            }
            case 6: {
                // With different covariance prescription
                gpz.setCovarianceType(CovarianceType::VARIABLE_LENGTH);
                break;
            }
            case 7: {
                // With different covariance prescription
                gpz.setCovarianceType(CovarianceType::VARIABLE_DIAGONAL);
                break;
            }
            case 8: {
                // With zero errors (see below)
                break;
            }
            case 9: {
                // With unity & uniform weights (see below)
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
            if (setup == 8) {
                errors.setZero(input.rows(), 1);
            }
            if (setup == 9) {
                weights.resize(input.rows());
                weights.fill(1.0);
            }

            gpz.fit(input, errors, output, weights);

            // Fetch model parameters
            GPzModel model = gpz.getModel();

            double ampMaxModel    = model.modelWeights[0];
            double centerPosModel = model.parameters.basisFunctionPositions(0,0)*model.featureSigma[0]
                                    +model.featureMean[0];
            double widthModel     = (1.0/model.parameters.basisFunctionCovariances[0](0,0))
                                    *model.featureSigma[0];
            double errorMinModel  = model.parameters.logUncertaintyConstant;
            double errorMaxModel  = model.parameters.uncertaintyBasisWeights[0];

            ampMaxStacked.accumulate(ampMaxModel);
            centerPosStacked.accumulate(centerPosModel);
            widthStacked.accumulate(widthModel);
            errorMinStacked.accumulate(errorMinModel);
            errorMaxStacked.accumulate(errorMaxModel);

            // Prepare testing data
            Mat2d inputTest;
            Vec1d testTrueValue;
            Vec1d testTrueUncertainty;
            generator.makeTesting(1000, inputTest, testTrueValue, testTrueUncertainty);

            // Do testing
            auto result = gpz.predict(inputTest, Mat2d{});

            if (s == 0) {
                trueValue = testTrueValue;
                trueUncertainty = testTrueUncertainty;
                if (debugTest) {
                    writeTesting("test.txt", inputTest, testTrueValue, testTrueUncertainty, result);
                }
            }

            // Fetch predicted values
            predValue.accumulate(result.value);
            predUncertainty.accumulate(result.uncertainty);
        }

        // Test that all attempts converged to a similar model (at better than 2%)
        BOOST_TEST(ampMaxStacked.relativeError()    < 0.02);
        BOOST_TEST(centerPosStacked.relativeError() < 0.02);
        BOOST_TEST(widthStacked.relativeError()     < 0.02);
        BOOST_TEST(errorMinStacked.relativeError()  < 0.02);
        BOOST_TEST(errorMaxStacked.relativeError()  < 0.02);

        // Test that all attempts converged to the right model (at better than 20%)
        double fudge = 10.0;
        BOOST_TEST(ampMaxStacked.mean()    == generator.ampMax,    tt::tolerance(fudge*ampMaxStacked.relativeError()));
        BOOST_TEST(centerPosStacked.mean() == generator.centerPos, tt::tolerance(fudge*centerPosStacked.relativeError()));
        BOOST_TEST(widthStacked.mean()     == generator.width,     tt::tolerance(fudge*widthStacked.relativeError()));
        BOOST_TEST(errorMinStacked.mean()  == generator.errorMin,  tt::tolerance(fudge*errorMinStacked.relativeError()));
        BOOST_TEST(errorMaxStacked.mean()  == generator.errorMax,  tt::tolerance(fudge*errorMaxStacked.relativeError()));

        // Test the predictions
        Vec1d meanValue = predValue.mean();
        Vec1d meanValueErr = predValue.relativeError();
        Vec1d meanUncertainty = predUncertainty.mean();
        Vec1d meanUncertaintyErr = predUncertainty.relativeError();

        fudge = 4.0;
        for (uint_t i = 0; i < trueValue.size(); ++i) {
            BOOST_TEST(meanValue[i] == trueValue[i], tt::tolerance(fudge*meanValueErr[i]));
            BOOST_TEST(meanUncertainty[i] == trueUncertainty[i], tt::tolerance(fudge*meanUncertaintyErr[i]));
        }
    }

}

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE_END ()


