#ifndef AEGEAN_CLUSTERING_OPT_PERSISTENCE_HPP
#define AEGEAN_CLUSTERING_OPT_PERSISTENCE_HPP

#include <limbo/tools/random_generator.hpp>

#include <Eigen/Dense>
#include <cassert>
#include <vector>

// Quick hack for definition of 'I' in <complex.h>
#undef I

namespace aegean {
    namespace clustering {
        namespace opt {

            template <typename ClusteringMethod, uint MIN_K, uint MAX_K>
            class Persistence {
            public:
                uint opt_k(const Eigen::MatrixXd& data, const uint reference_replicates = 25)
                {
                    assert(MIN_K > 1);

                    Eigen::VectorXd betas(MAX_K - MIN_K + 2);
                    for (uint idx = 0, k = MIN_K - 1; k < MAX_K + 1; ++idx, ++k) {
                        _cm.fit(data, static_cast<int>(k));
                        Eigen::VectorXi labels = _cm.labels();
                        Eigen::MatrixXd centroids = _cm.centroids();

                        Eigen::VectorXd lambdas(centroids.rows());
                        for (uint i = 0; i < centroids.rows(); ++i) {
                            Eigen::MatrixXd diff = data;
                            for (uint r = 0; r < data.rows(); ++r)
                                diff.row(r) -= centroids.row(i);

                            Eigen::VectorXi v = (labels.array() == i).select(Eigen::VectorXi::Ones(labels.rows(), labels.cols()), 0);
                            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(diff.cols(), diff.cols());
                            for (uint r = 0; r < diff.rows(); ++r)
                                C += (diff.row(r).transpose() * diff.row(r)) * v(r);
                            lambdas(i) = C.eigenvalues().real().array().maxCoeff();
                        }
                        betas(idx) = 1. / (2 * lambdas.maxCoeff());
                    }

                    Eigen::VectorXd logdiffs(betas.rows() - 1);
                    for (uint i = 1; i < logdiffs.rows(); ++i) {
                        logdiffs(i) = std::log(betas(i)) - std::log(betas(i - 1));
                    }

                    Eigen::MatrixXd::Index maxIdx;
                    logdiffs.maxCoeff(&maxIdx);
                    return MIN_K + maxIdx;
                }

            protected:
                ClusteringMethod _cm;
            };

        } // namespace opt
    } // namespace clustering
} // namespace aegean

#endif