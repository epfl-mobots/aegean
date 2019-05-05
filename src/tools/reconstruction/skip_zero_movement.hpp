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

                    Eigen::MatrixXd filtered;
                    Eigen::MatrixXd last_row = matrix.row(0);
                    for (uint i = 1; i < matrix.rows(); ++i) {
                        Eigen::MatrixXd dif = last_row.array() - matrix.row(i).array();
                        double mse = dif.norm();
                        last_row = matrix.row(i);

                        if (mse <= Params::SkipZeroMovement::eps)
                            continue;

                        filtered.conservativeResize(filtered.rows() + 1, matrix.cols());
                        filtered.row(filtered.rows() - 1) = matrix.row(i);
                    }

                    std::cout << "Lines skipped: " << matrix.rows() - filtered.rows() << " (out of " << matrix.rows() << ")" << std::endl;
                    matrix = filtered;
                }
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif