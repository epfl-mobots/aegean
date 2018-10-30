#ifndef AEGEAN_TOOLS_RECONSTRUCTION_CSPACE_HPP
#define AEGEAN_TOOLS_RECONSTRUCTION_CSPACE_HPP

#include "reconstruction_base.hpp"
#include <tools/mathtools.hpp>
#include <tools/polygons/circular_corridor.hpp>

#include <Eigen/Core>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace aegean {
    namespace tools {
        namespace reconstruction {

            using namespace polygons;

            template <typename CircularCorridor>
            struct CSpace : public ReconstructionBase {
            public:
                void operator()(Eigen::MatrixXd& matrix) override
                {
                    for (uint i = 0; i < matrix.cols(); i += 2) {
                        Eigen::MatrixXd individual = matrix.block(0, i, matrix.rows(), 2);

                        // check which rows are valid to help us reconstruct the missing
                        // value intervals
                        std::vector<uint> rows_wo_nans;
                        for (uint j = 0; j < individual.rows(); ++j) {
                            if (individual.row(j).array().isNaN().count() == 0)
                                rows_wo_nans.push_back(j);
                        }

                        // reconstruct intervals first
                        for (uint j = 0; j < rows_wo_nans.size() - 1; ++j) {
                            uint current = rows_wo_nans[j];
                            uint next = rows_wo_nans[j + 1];
                            uint num_missing = next - current;
                            if (num_missing > 1) {
                                _generate_circular_trajectory(individual, num_missing, current,
                                    next, true);
                            }
                        }

                        // reconstruct missing values in the beginning and end of thetrajectories if
                        // (rows_wo_nans[0] > 0)
                        _generate_circular_trajectory(individual, rows_wo_nans[0], rows_wo_nans[2],
                            rows_wo_nans[1], false);

                        if (rows_wo_nans[rows_wo_nans.size() - 1] < matrix.rows() - 1) {
                            int num_missing
                                = abs(static_cast<int>(matrix.rows())
                                      - static_cast<int>(rows_wo_nans[rows_wo_nans.size() - 1]))
                                - 1;

                            _generate_circular_trajectory(
                                individual, num_missing, rows_wo_nans[rows_wo_nans.size() - 2],
                                rows_wo_nans[rows_wo_nans.size() - 1], false);
                        }

                        // do a final check to validate that all cells have been reconstructed
                        rows_wo_nans.clear();
                        for (uint j = 0; j < individual.rows(); ++j) {
                            if (individual.row(j).array().isNaN().count() == 0)
                                rows_wo_nans.push_back(j);
                        }

                        assert((matrix.rows() == static_cast<uint>(rows_wo_nans.size()))
                            && "Reconstructing went wrong");
                        matrix.col(i) = individual.col(0);
                        matrix.col(i + 1) = individual.col(1);
                    }
                }

            protected:
                void _generate_circular_trajectory(Eigen::MatrixXd& individual,
                    const uint num_missing,
                    const uint first_valid_idx,
                    const uint second_valid_idx,
                    const bool fit_between)
                {
                    double r;
                    std::tuple<double, double, double> thetas;
                    std::tie(r, thetas)
                        = _fit_circle(individual, first_valid_idx, second_valid_idx);

                    const int sign = sgn(std::get<0>(thetas));
                    double angle;
                    double phi;

                    if (fit_between) {
                        angle = std::get<1>(thetas);
                        phi = std::fabs(std::get<0>(thetas)) / num_missing;

                        for (uint coef = 1, i = first_valid_idx + 1; i < second_valid_idx;
                             ++coef, ++i) {

                            individual(i, 0)
                                = r * std::cos((angle - sign * phi * coef) * M_PI / 180)
                                + _cc.center().x();

                            individual(i, 1)
                                = r * std::sin((angle - sign * phi * coef) * M_PI / 180)
                                + _cc.center().y();
                        }
                    }
                    else {
                        angle = std::get<2>(thetas);
                        phi = std::fabs(std::get<0>(thetas));

                        int direction = sgn(static_cast<int>(first_valid_idx)
                            - static_cast<int>(second_valid_idx));
                        int start_idx;
                        (direction < 0) ? start_idx = first_valid_idx - direction
                                        : start_idx = second_valid_idx - direction;

                        for (uint coef = 1, i = 0; i < num_missing; ++coef, ++i) {
                            uint idx = start_idx - direction * (i + 1);

                            individual(idx, 0)
                                = r * std::cos((angle - sign * phi * coef) * M_PI / 180)
                                + _cc.center().x();

                            individual(idx, 1)
                                = r * std::sin((angle - sign * phi * coef) * M_PI / 180)
                                + _cc.center().y();
                        }
                    }
                }

                std::pair<double, std::tuple<double, double, double>>
                _fit_circle(const Eigen::MatrixXd& individual, uint first_valid_idx,
                    uint second_valid_idx) const
                {
                    Eigen::Vector2d r1, r2;
                    r1(0) = individual(first_valid_idx, 0) - _cc.center().x();
                    r1(1) = individual(first_valid_idx, 1) - _cc.center().y();
                    r2(0) = individual(second_valid_idx, 0) - _cc.center().x();
                    r2(1) = individual(second_valid_idx, 1) - _cc.center().y();
                    double r = (r1.norm() + r2.norm()) / 2;

                    double theta1 = std::atan2(individual(first_valid_idx, 1) - _cc.center().y(),
                                        individual(first_valid_idx, 0) - _cc.center().x())
                        * 180 / M_PI;
                    theta1 = (theta1 < 0) ? theta1 + 360 : theta1;
                    double theta2 = std::atan2(individual(second_valid_idx, 1) - _cc.center().y(),
                                        individual(second_valid_idx, 0) - _cc.center().x())
                        * 180 / M_PI;
                    theta2 = (theta2 < 0) ? theta2 + 360 : theta2;
                    double theta = theta1 - theta2;
                    if (std::fabs(theta) > 180.0f)
                        theta = 360 - std::fabs(theta);
                    return std::make_pair(r, std::make_tuple(theta, theta1, theta2));
                }

                CircularCorridor _cc;
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif