#ifndef AEGEAN_FEATURES_VELOCITY_HPP
#define AEGEAN_FEATURES_VELOCITY_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class Velocity : public FeatureBase {
        public:
            Velocity() : _feature_name("velocity") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(matrix.cols()) * 0.01;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = matrix(0, i) + noise(i);
                Eigen::MatrixXd velocities = (matrix - rolled) / timestep;
                _velocity = velocities.array().abs().rowwise().mean();
            }

            Eigen::MatrixXd get() override { return _velocity; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _velocity;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif