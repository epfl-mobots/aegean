#ifndef AEGEAN_FEATURES_BINARY_POLARIZATION_HPP
#define AEGEAN_FEATURES_BINARY_POLARIZATION_HPP

#include <cmath>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        template <typename CircularCorridor>
        class BinaryPolarization : public FeatureBase {
        public:
            BinaryPolarization() : _feature_name("binary_polarization") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                CircularCorridor cc;
                uint duration = matrix.rows();
                Eigen::MatrixXd angles(matrix.rows(), matrix.cols() / 2);
                for (uint ind = 0; ind < matrix.cols() / 2; ++ind) {
                    Eigen::MatrixXd x_centered = matrix.col(ind * 2).array() - cc.center().x();
                    Eigen::MatrixXd y_centered = matrix.col(ind * 2 + 1).array() - cc.center().y();
                    for (uint i = 0; i < duration; ++i)
                        angles(i, ind) = std::fmod(std::atan2(y_centered(i), x_centered(i)) * 180 / M_PI + 360, 360);
                }

                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(angles.cols()) * 3;
                Eigen::MatrixXd rolled = tools::rollMatrix(angles, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = angles(0, i) + noise(i);
                Eigen::MatrixXd grad = (angles - rolled) / timestep;

                Eigen::MatrixXd polarization(angles.rows(), angles.cols());
                for (uint ind = 0; ind < matrix.cols() / 2; ++ind) {
                    for (uint i = 0; i < duration; ++i) {
                        if (grad(i, ind) > 0) // if grad positive we are moving clockwise
                            polarization(i, ind) = -1; // -1 for clockwise
                        else if (grad(i, ind) < 0) // if grad negative we are moving counter-clockwise
                            polarization(i, ind) = 1; // 1 for counter-clockwise
                        else
                            (i > 0) ? polarization(i, ind) = polarization(i - 1, ind) : polarization(i, ind) = -1;

                        // need to check the transition between 0-360 and vice versa
                        // very hacky but should work
                        if (i > 0) {
                            if (angles(i - 1, ind) > 270 && angles(i, ind) < 90) {
                                polarization(i, ind) = -1;
                            }

                            if (angles(i - 1, ind) < 90 && angles(i, ind) > 270) {
                                polarization(i, ind) = 1;
                            }
                        }
                    }
                }

                _binary_polarization = polarization.rowwise().mean();
            }

            Eigen::MatrixXd get() override { return _binary_polarization; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _binary_polarization;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif