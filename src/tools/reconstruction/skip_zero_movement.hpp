#ifndef AEGEAN_TOOLS_RECONSTRUCTION_SKIP_ZERO_MOVEMENT_HPP
#define AEGEAN_TOOLS_RECONSTRUCTION_SKIP_ZERO_MOVEMENT_HPP

#include "last_known.hpp"
#include <Eigen/Core>

#include <iostream>

namespace aegean {
    namespace defaults {
        struct SkipZeroMovement {
            static constexpr double eps = 0.0005;
        };
    } // namespace defaults

    namespace tools {
        namespace reconstruction {

            template <typename Params>
            struct SkipZeroMovement : public LastKnown {
                void operator()(Eigen::MatrixXd& matrix) override
                {
                    LastKnown::operator()(matrix);

                    int zero_movement;
                    Eigen::MatrixXd reference = matrix;
                    do {
                        zero_movement = 0;
                        Eigen::MatrixXd filtered;
                        Eigen::MatrixXd last_row = reference.row(0);
                        for (uint i = 1; i < reference.rows(); ++i) {
                            Eigen::MatrixXd dif = last_row.array() - reference.row(i).array();
                            double mse = dif.norm();
                            last_row = reference.row(i);

                            if (mse <= Params::SkipZeroMovement::eps) {
                                ++zero_movement;
                                continue;
                            }

                            filtered.conservativeResize(filtered.rows() + 1, reference.cols());
                            filtered.row(filtered.rows() - 1) = reference.row(i);
                        }

                        reference = filtered;
                    } while (zero_movement > 0);

                    std::cout << "Lines skipped: " << matrix.rows() - reference.rows() << " (out of " << matrix.rows() << ")" << std::endl;
                    matrix = reference;
                }
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif