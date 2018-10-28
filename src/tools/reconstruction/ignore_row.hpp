#ifndef AEGEAN_TOOLS_RECONSTRUCTION_IGNORE_ROW_HPP
#define AEGEAN_TOOLS_RECONSTRUCTION_IGNORE_ROW_HPP

#include "reconstruction_base.hpp"
#include <Eigen/Core>

#include <iostream>

namespace aegean {
    namespace tools {
        namespace reconstruction {
            struct IgnoreRow : public ReconstructionBase {
                void operator()(Eigen::MatrixXd& matrix) override
                {
                    Eigen::MatrixXd filtered;
                    for (uint i = 0; i < matrix.rows(); ++i) {
                        Eigen::MatrixXd row = matrix.row(i);
                        if (row.array().isNaN().count() > 0) {
                            filtered.conservativeResize(filtered.rows() + 1, matrix.cols());
                            filtered.row(filtered.rows() - 1) = row;
                        }
                    }
                    matrix = filtered;
                }
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif