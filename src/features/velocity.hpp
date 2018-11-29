#ifndef AEGEAN_FEATURES_VELOCITY_HPP
#define AEGEAN_FEATURES_VELOCITY_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class Velocity : public FeatureBase {
        public:
            Velocity() : _feature_name("velocity") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                Eigen::VectorXd noise = Eigen::VectorXd::Ones(matrix.cols())
                    - limbo::tools::random_vector_bounded(matrix.cols()) / 10;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, -1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(duration - 1, i) = matrix(duration - 1, i) * noise(i);
                Eigen::MatrixXd velocities = (rolled - matrix) / timestep;
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