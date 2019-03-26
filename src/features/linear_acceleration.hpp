#ifndef AEGEAN_FEATURES_LINEAR_ACCELERATION_HPP
#define AEGEAN_FEATURES_LINEAR_ACCELERATION_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class LinearAcceleration : public FeatureBase {
        public:
            LinearAcceleration() : _feature_name("linear_acceleration") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(matrix.cols()) * 0.01;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = matrix(0, i) + noise(i);
                Eigen::MatrixXd velocities = (matrix - rolled) / timestep;

                noise = limbo::tools::random_vector_bounded(velocities.cols()) * 0.001;
                rolled = tools::rollMatrix(velocities, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = velocities(0, i) + noise(i);
                Eigen::MatrixXd accelerations = (velocities - rolled) / timestep;

                _linear_acceleration = Eigen::MatrixXd(duration, matrix.cols() / 2);
                for (uint r = 0; r < _linear_acceleration.rows(); ++r) {
                    for (uint c = 0; c < _linear_acceleration.cols(); ++c) {
                        double phi = std::atan2(accelerations(r, c * 2 + 1), accelerations(r, c * 2));
                        double resultant = std::sqrt(
                            std::pow(accelerations(r, c * 2), 2)
                            + std::pow(accelerations(r, c * 2 + 1), 2)
                            + 2 * accelerations(r, c * 2) * accelerations(r, c * 2)
                                * std::cos(phi));
                        _linear_acceleration(r, c) = resultant;
                    }
                }
            }

            Eigen::MatrixXd get() override { return _linear_acceleration; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _linear_acceleration;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif