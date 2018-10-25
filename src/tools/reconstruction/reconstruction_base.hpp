#ifndef FILTERING_BASE_HPP
#define FILTERING_BASE_HPP

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