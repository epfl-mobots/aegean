#ifndef AEGEAN_TOOLS_RECONSTRUCTION_LAST_KNOWN_HPP
#define AEGEAN_TOOLS_RECONSTRUCTION_LAST_KNOWN_HPP

#include "reconstruction_base.hpp"
#include <Eigen/Core>

#include <iostream>

namespace aegean {
    namespace tools {
        namespace reconstruction {
            struct LastKnown : public ReconstructionBase {
                void operator()(Eigen::MatrixXd& matrix) override
                {
                    Eigen::MatrixXd filtered;
                    uint last_valid_idx = -1;
                    for (uint i = 0; i < matrix.rows(); ++i) {
                        uint idx = i;
                        Eigen::MatrixXd row = matrix.row(i);
                        if (row.array().isNaN().count() > 0)
                            last_valid_idx = i;
                        else if (last_valid_idx > 0)
                            idx = last_valid_idx;
                        filtered.conservativeResize(filtered.rows() + 1, matrix.cols());
                        filtered.row(filtered.rows() - 1) = matrix.row(idx);
                    }
                    matrix = filtered;
                }
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif