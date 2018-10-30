#ifndef AEGEAN_CLUSTERING_OPT_NO_OPT_HPP
#define AEGEAN_CLUSTERING_OPT_NO_OPT_HPP

#include <Eigen/Core>
#include <vector>

namespace aegean {
    namespace clustering {
        namespace opt {

            template <uint K>
            class NoOpt {
            public:
                uint opt_k(const Eigen::MatrixXd& data)
                {
                    return K;
                }
            };

        } // namespace opt
    } // namespace clustering
} // namespace aegean

#endif