#ifndef BEARING_HPP
#define BEARING_HPP

#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>
#include <cmath>

#include <iostream>

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
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, -1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(duration - 1, i) = matrix(duration - 1, i) * noise(i);
                Eigen::MatrixXd velocities = (rolled - matrix) / timestep;

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

          private:
            std::string _feature_name;
            Eigen::MatrixXd _bearing;
        };

    } // namespace features
} // namespace aegean

#endif