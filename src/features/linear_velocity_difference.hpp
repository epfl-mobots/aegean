#ifndef AEGEAN_FEATURES_LINEAR_VELOCITY_DIFFERENCE_HPP
#define AEGEAN_FEATURES_LINEAR_VELOCITY_DIFFERENCE_HPP

#include <cmath>
#include <features/linear_velocity.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

#include <iostream>

namespace aegean {
    namespace features {

        class LinearVelocityDifference : public LinearVelocity {
        public:
            LinearVelocityDifference() : _feature_name("linear_velocity_difference") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                uint duration = matrix.rows();
                LinearVelocity::operator()(matrix, timestep);

                _linear_velocity_diff.resize(matrix.cols() / 2, Eigen::MatrixXd::Zero(duration, matrix.cols() / 2));
                for (uint r = 0; r < duration; ++r) {
                    for (uint k = 0; k < matrix.cols() / 2; ++k) {
                        for (uint l = 0; l < matrix.cols() / 2; ++l) {
                            if (k == l) // k is the focal individual
                                continue;
                            _linear_velocity_diff[k](r, l) = _linear_velocity(r, k) - _linear_velocity(r, l);
                        } // l
                    } // k
                } // r
            }

            const std::vector<Eigen::MatrixXd>& get_vec() override { return _linear_velocity_diff; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            std::vector<Eigen::MatrixXd> _linear_velocity_diff;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif