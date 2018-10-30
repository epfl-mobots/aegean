#ifndef AEGEAN_CLUSTERING_OPT_GAP_STATISTIC_HPP
#define AEGEAN_CLUSTERING_OPT_GAP_STATISTIC_HPP

#include <tools/random_generator.hpp>

#include <Eigen/Core>
#include <vector>

// Quick hack for definition of 'I' in <complex.h>
#undef I

namespace aegean {
    namespace opt {

        template <typename ClusteringMethod>
        class GapStatistic {
        public:
            uint opt_k(const Eigen::MatrixXd& data, const uint min_k, const uint max_k, const uint reference_replicates = 10)
            {
                const Eigen::MatrixXd bbox = _bbox(data);

                Eigen::VectorXd gaps = Eigen::VectorXd::Zero(max_k - min_k + 1);
                Eigen::VectorXd errors = Eigen::VectorXd::Zero(max_k - min_k + 1);
                for (uint k = min_k; k < max_k + 1; ++k) {
                    Eigen::VectorXd inertia = Eigen::VectorXd::Zero(reference_replicates);
                    for (uint nr = 0; nr < reference_replicates; ++nr) {
                        Eigen::MatrixXd reference = _generate_reference(bbox, data.rows(), data.cols());
                        _cm.fit(reference, static_cast<int>(k));
                        inertia(nr) = _cm.inertia();
                    }
                    _cm.fit(data, k);
                    double orig_inertia = _cm.inertia();
                    double std_dev = std::sqrt((inertia.array() - inertia.mean()).square().sum() / inertia.size());
                    errors(k - min_k) = std::sqrt(1 + 1 / reference_replicates) * std_dev;
                    gaps(k - min_k) = std::log(inertia.mean()) - std::log(orig_inertia);
                }

                for (uint i = 0; i < gaps.rows() - 1; ++i) {
                    if (gaps(i) > gaps(i + 1))
                        return i + min_k;
                }
                return max_k;
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
                Eigen::MatrixXd ref(rows, cols);
                for (uint i = 0; i < cols; ++i) {
                    limbo::tools::rgen_double_t rgen(bbox(0, i), bbox(1, i));
                    for (uint j = 0; j < rows; ++j) {
                        ref(j, i) = rgen.rand();
                    }
                }
                return ref;
            }

            ClusteringMethod _cm;
        };

    } // namespace opt
} // namespace aegean

#endif