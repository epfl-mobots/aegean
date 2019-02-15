#ifndef AEGEAN_FEATURES_ANGULAR_VELOCITY_DIFFERENCE_HPP
#define AEGEAN_FEATURES_ANGULAR_VELOCITY_DIFFERENCE_HPP

#include <cmath>
#include <features/angular_velocity.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

#include <iostream>

namespace aegean {
    namespace features {

        class AngularVelocityDifference : public AngularVelocity {
        public:
            AngularVelocityDifference() : _feature_name("angular_velocity_difference") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                using namespace tools;

                uint duration = matrix.rows();
                AngularVelocity::operator()(matrix, timestep);
                _angular_velocity_diff.resize(matrix.cols() / 2, Eigen::MatrixXd::Zero(duration, matrix.cols() / 2));
                for (uint r = 0; r < duration; ++r) {
                    for (uint k = 0; k < matrix.cols() / 2; ++k) {
                        for (uint l = 0; l < matrix.cols() / 2; ++l) {
                            if (k == l) // k is the focal individual
                                continue;
                            _angular_velocity_diff[k](r, l) = -(_angular_velocity(r, k) - _angular_velocity(r, l)) / timestep;
                        } // l
                    } // k
                } // r
            }

            const std::vector<Eigen::MatrixXd>& get_vec() override { return _angular_velocity_diff; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            std::vector<Eigen::MatrixXd> _angular_velocity_diff;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif