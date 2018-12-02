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
#include <Eigen/Dense>

namespace PHZ_GPz {
	using Mat2d = Eigen::MatrixXd;
	using Vec2d = Eigen::ArrayXXd;
	using Vec1d = Eigen::VectorXd;

    class {
    public:
        
        enum class PriorMeanFunction {
            ZERO, LINEAR
        };
        
        enum class CovarianceType {
            GLOBAL_LENGTH,      // GPGL
            VARIABLE_LENGTH,    // GPVL
            GLOBAL_DIAGONAL,    // GPGD
            VARIABLE_DIAGONAL,  // GPVD
            GLOBAL_COVARIANCE,  // GPGC
            VARIABLE_COVARIANCE // GPVC
        };
    
    private:
    
        // Configuration
        std::size_t numberPseudoInputs_ = 100;
        PriorMeanFunction priorMean_ = PriorMeanFunction::LINEAR;
        CovarianceType covarianceType_ = CovarianceType::VARIABLE_COVARIANCE;
        
        
        // Internal variables
        std::size_t numberFeatures_ = 0;
        std::size_t numberParameters_ = 0;
        
        // Training state
        Vec1d parameters_;
        
    private:
        
        void updateNumberParameters_() {
            numberParameters_ = numberPseudoInputs_;
            
            switch (covarianceType_) {
                case GLOBAL_LENGTH: {
                    numberParameters_ += 1;
                    break;
                }
                case VARIABLE_LENGTH: {
                    numberParameters_ += numberPseudoInputs_;
                    break;
                }
                case GLOBAL_DIAGONAL: {
                    numberParameters_ += numberFeatures_;
                    break;
                }
                case VARIABLE_DIAGONAL : {
                    numberParameters_ += numberFeatures_*numberPseudoInputs_;
                    break;
                }
                case GLOBAL_COVARIANCE : {
                    numberParameters_ += numberFeatures_*numberFeatures_;
                    break;
                }
                case VARIABLE_COVARIANCE : {
                    numberParameters_ += numberFeatures_*numberFeatures_*numberPseudoInputs_;
                    break;
                }
            }
            
            switch (priorMean_) {
                case ZERO: {
                    break;
                }
                case LINEAR: {
                    numberParameters_ += numberFeatures_ + 1;
                    break;
                }
            };
        }
        
        void updateParameters_() {
            updateNumberParameters_();
            
            parameters_.resize(numberParameters_);
            
            // Set sensible starting values
        }
        
    public:
    
        GPz()                      = default;
        ~GPz()                     = default;        
        GPz(const GPz&)            = default;
        GPz(GPz&&)                 = default;
        GPz& operator=(const GPz&) = default;
        GPz& operator=(GPz&&)      = default;
    
        std::size_t getNumberOfPseudoInputs() const {
            return numberPseudoInputs_;
        }
        
        void setNumberOfPseudoInputs(std::size_t num) {
            if (num != numberPseudoInputs_) {
                numberPseudoInputs_ = num;
                updateCache_ = true;
            }
        }
        
        void setPriorMeanFunction(PriorMeanFunction newFunction) {
            if (newFunction != priorMean_) {
                priorMean_ = newFunction;
                updateCache_ = true;
            }
        }
    
        PriorMeanFunction getPriorMeanFunction() const {
            return priorMean_;
        }
    
        void fit(const Vec2d& x, const Vec2d& xe, const Vec1d& y) {
        }
        
        const Vec1d& get_parameters() const {
            return parameters;
        }
        
        void set_parameters(Vec1d newParameters) {
            // Check that the provided vector has the right size
            updateNumberParameters_();
            assert(new_parameters.size() == numberParameters_);
            
            // Assign
            parameters_ = std::move(newParameters);
        }
    };
}  // namespace PHZ_GPz



