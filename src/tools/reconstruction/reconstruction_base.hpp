#ifndef AEGEAN_TOOLS_RECONSTRUCTION_FILTERING_BASE_HPP
#define AEGEAN_TOOLS_RECONSTRUCTION_FILTERING_BASE_HPP

#include <Eigen/Core>
#include <cassert>

namespace aegean {
    namespace tools {
        namespace reconstruction {
            struct ReconstructionBase {
                virtual void operator()(Eigen::MatrixXd& matrix) { assert(false); }
            };

        } // namespace reconstruction
    } // namespace tools
} // namespace aegean

#endif