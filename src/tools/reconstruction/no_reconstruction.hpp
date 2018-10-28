#ifndef AEGEAN_TOOLS_RECONSTRUCTION_NO_FILTERING_HPP
#define AEGEAN_TOOLS_RECONSTRUCTION_NO_FILTERING_HPP

#include "reconstruction_base.hpp"
#include <Eigen/Core>

namespace aegean {
    namespace tools {
        namespace reconstruction {
            struct NoReconstruction : public ReconstructionBase {
                void operator()(Eigen::MatrixXd& matrix) override {}
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif