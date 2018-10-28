#ifndef AEGEAN_FEATURES_INTER_INDIVIDUAL_DISTANCE_HPP
#define AEGEAN_FEATURES_INTER_INDIVIDUAL_DISTANCE_HPP

#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>
#include <cmath>

#include <iostream>

namespace aegean {
    namespace defaults {
        namespace distance_functions {

            template <typename CircularCorridor>
            struct angular {
                double operator()(Eigen::VectorXd pt1, Eigen::VectorXd pt2) const
                {
                    pt1(0) -= _cc.center().x();
                    pt1(1) -= _cc.center().y();
                    pt2(0) -= _cc.center().x();
                    pt2(1) -= _cc.center().y();

                    double theta1 = std::fmod(std::atan2(pt1(1), pt1(0)) * 180 / M_PI + 360, 360);
                    double theta2 = std::fmod(std::atan2(pt2(1), pt2(0)) * 180 / M_PI + 360, 360);
                    double theta = fabs(theta1 - theta2);
                    return (theta > 180) ? 360 - theta : theta;
                }

                CircularCorridor _cc;
            };

            struct euclidean {
                double operator()(Eigen::VectorXd pt1, Eigen::VectorXd pt2) const
                {
                    return (pt1 - pt2).norm();
                }
            };

        } // namespace distance_functions
    } // namespace defaults

    namespace features {

        template <typename DistanceFunc>
        class InterIndividualDistance : public FeatureBase {
          public:
            InterIndividualDistance() : _feature_name("InterIndividual") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                _iid = Eigen::MatrixXd::Zero(duration, 1);
                for (uint i = 0; i < matrix.rows(); ++i) {
                    for (uint j = 0; j < matrix.cols() / 2; ++j) {
                        double focal_sum = 0;
                        for (uint k = 0; k < matrix.cols() / 2; ++k) {
                            if (j == k)
                                continue; // skipping distance of individual withitself
                            Eigen::MatrixXd focal(2, 1), neigh(2, 1);
                            focal(0) = matrix(i, j * 2);
                            focal(1) = matrix(i, j * 2 + 1);
                            neigh(0) = matrix(i, k * 2);
                            neigh(1) = matrix(i, k * 2 + 1);
                            focal_sum += _distance(focal, neigh);
                        }
                        _iid(i) += focal_sum / (matrix.cols() / 2 - 1);
                    }
                    _iid(i) /= (matrix.cols() / 2);
                }
                _iid /= 360;
            }

            Eigen::MatrixXd get() override { return _iid; }

            const std::string& feature_name() override { return _feature_name; }

          protected:
            Eigen::MatrixXd _iid;

          private:
            DistanceFunc _distance;
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif