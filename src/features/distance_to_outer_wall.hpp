#ifndef AEGEAN_FEATURES_DISTANCE_TO_OUTER_WALL_HPP
#define AEGEAN_FEATURES_DISTANCE_TO_OUTER_WALL_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>
#include <tools/primitives/point.hpp>

#include <iostream>

namespace aegean {
    namespace features {

        template <typename CircularSetup>
        class DistanceToOuterWall : public FeatureBase {
        public:
            DistanceToOuterWall() : _feature_name("distance_to_outer_wall") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                _dtw = Eigen::MatrixXd::Zero(matrix.rows(), 1);
                for (uint i = 0; i < matrix.rows(); ++i) {
                    double sum_ts_distance = 0.;
                    for (uint j = 0; j < matrix.cols() / 2; ++j) {
                        tools::primitives::Point p;
                        p.x() = matrix(i, j * 2);
                        p.y() = matrix(i, j * 2 + 1);
                        sum_ts_distance += _setup.distance_to_outer_wall(p);
                    }
                    _dtw(i) = sum_ts_distance / (matrix.cols() / 2);
                }
            }

            Eigen::MatrixXd get() override { return _dtw; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            CircularSetup _setup;
            Eigen::MatrixXd _dtw;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif