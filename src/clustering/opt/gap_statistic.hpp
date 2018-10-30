#ifndef AEGEAN_CLUSTERING_OPT_GAP_STATISTIC_HPP
#define AEGEAN_CLUSTERING_OPT_GAP_STATISTIC_HPP

#include <tools/random_generator.hpp>

#include <Eigen/Core>
#include <vector>

// Quick hack for definition of 'I' in <complex.h>
#undef I

namespace aegean {
    namespace clustering {
        namespace opt {

            template <typename ClusteringMethod, uint MIN_K, uint MAX_K>
            class GapStatistic {
            public:
                uint opt_k(const Eigen::MatrixXd& data, const uint reference_replicates = 25)
                {
                    const Eigen::MatrixXd bbox = _bbox(data);

                    Eigen::VectorXd gaps = Eigen::VectorXd::Zero(MAX_K - MIN_K + 1);
                    Eigen::VectorXd errors = Eigen::VectorXd::Zero(MAX_K - MIN_K + 1);
                    for (uint k = MIN_K; k < MAX_K + 1; ++k) {
                        Eigen::VectorXd inertia = Eigen::VectorXd::Zero(reference_replicates);
                        for (uint nr = 0; nr < reference_replicates; ++nr) {
                            Eigen::MatrixXd reference = _generate_reference(bbox, data.rows(), data.cols());
                            _cm.fit(reference, static_cast<int>(k));
                            inertia(nr) = _cm.inertia();
                        }
                        _cm.fit(data, static_cast<int>(k));
                        double orig_inertia = _cm.inertia();
                        double std_dev = std::sqrt((inertia.array() - inertia.mean()).square().sum() / inertia.size());
                        errors(k - MIN_K) = std::sqrt(1 + 1 / reference_replicates) * std_dev;
                        gaps(k - MIN_K) = std::log(inertia.mean()) - std::log(orig_inertia);
                    }

                    for (uint i = 0; i < gaps.rows() - 1; ++i) {
                        if (gaps(i) > gaps(i + 1))
                            return i + MIN_K;
                    }
                    return MAX_K;
                }

            protected:
                const Eigen::MatrixXd _bbox(const Eigen::MatrixXd& data) const
                {
                    Eigen::MatrixXd bbox(2, data.cols());
                    for (uint i = 0; i < data.cols(); ++i) {
                        bbox(0, i) = data.col(i).minCoeff();
                        bbox(1, i) = data.col(i).maxCoeff();
                    }
                    return bbox;
                }

                const Eigen::MatrixXd _generate_reference(const Eigen::MatrixXd& bbox, uint rows, uint cols) const
                {
                    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(rows, cols);
                    for (uint i = 0; i < cols; ++i) {
                        limbo::tools::rgen_double_t rgen(bbox(0, i), bbox(1, i));
                        ref.col(i) = ref.col(i).unaryExpr([&](double val) { return rgen.rand(); });
                    }
                    return ref;
                }

                ClusteringMethod _cm;
            };

        } // namespace opt
    } // namespace clustering
} // namespace aegean

#endif