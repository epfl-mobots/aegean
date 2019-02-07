#ifndef AEGEAN_FEATURES_BEARING_HPP
#define AEGEAN_FEATURES_BEARING_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class Bearing : public FeatureBase {
        public:
            Bearing() : _feature_name("bearing") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                Eigen::VectorXd noise = Eigen::VectorXd::Ones(matrix.cols())
                    - limbo::tools::random_vector_bounded(matrix.cols()) / 10;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = matrix(0, i) * noise(i);
                Eigen::MatrixXd velocities = (matrix - rolled) / timestep;

                _bearing.resize(matrix.rows(), matrix.cols() / 2);
                for (uint i = 0; i < _bearing.rows(); ++i) {
                    for (uint j = 0; j < _bearing.cols(); ++j) {
                        _bearing(i, j) = std::atan2(velocities(i, j * 2 + 1), velocities(i, j * 2))
                            * 180.0f / M_PI;
                        _bearing(i, j) = std::fmod(_bearing(i, j) + 360, 360);
                    }
                }
            }

            Eigen::MatrixXd get() override { return _bearing; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _bearing;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif