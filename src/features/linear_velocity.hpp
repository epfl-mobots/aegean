#ifndef AEGEAN_FEATURES_LINEAR_VELOCITY_HPP
#define AEGEAN_FEATURES_LINEAR_VELOCITY_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class LinearVelocity : public FeatureBase {
        public:
            LinearVelocity() : _feature_name("linear_velocity") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(matrix.cols()) * 0.01;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = matrix(0, i) + noise(i);
                Eigen::MatrixXd velocities = (matrix - rolled) / timestep;

                _linear_velocity = Eigen::MatrixXd(duration, matrix.cols() / 2);
                for (uint r = 0; r < _linear_velocity.rows(); ++r) {
                    for (uint c = 0; c < _linear_velocity.cols(); ++c) {
                        double phi = std::atan2(velocities(r, c * 2 + 1), velocities(r, c * 2));
                        double resultant = std::sqrt(
                            std::pow(velocities(r, c * 2), 2)
                            + std::pow(velocities(r, c * 2 + 1), 2)
                            + 2 * velocities(r, c * 2) * velocities(r, c * 2)
                                * std::sin(phi));
                        _linear_velocity(r, c) = resultant;
                    }
                }
            }

            Eigen::MatrixXd get() override { return _linear_velocity; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _linear_velocity;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif