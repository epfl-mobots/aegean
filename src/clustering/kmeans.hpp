#ifndef AEGEAN_CLUSTERING_KMEANS_HPP
#define AEGEAN_CLUSTERING_KMEANS_HPP

#include <vector>
#include <Eigen/Core>

// Quick hack for definition of 'I' in <complex.h>
#undef I

#include <tools/random_generator.hpp>

namespace aegean {
    namespace clustering {
        class KMeans {
          public:
            KMeans() {}

            std::vector<Eigen::MatrixXd> fit(const Eigen::MatrixXd& data, const int K,
                                             const int num_init = 5, const int max_iter = 100)
            {
                static thread_local limbo::tools::rgen_int_t rgen(0, data.rows() - 1);

                _centroids = Eigen::MatrixXd::Zero(K, data.cols());
                _labels = Eigen::VectorXi::Ones(data.rows()) * -1;

                // random points as centroids in the beginning
                for (int i = 0; i < K; ++i) {
                    _centroids.row(i) = data.row(rgen.rand());
                }

                Eigen::MatrixXd prev_centroids;
                for (int n = 0; n < max_iter; ++n) {
                    // assign points to cluster
                    _clusters.clear();
                    _clusters.resize(K);
                    _inertia = 0;

                    for (int i = 0; i < data.rows(); ++i) {
                        double min = std::numeric_limits<double>::max();
                        int min_k = -1;
                        for (int k = 0; k < K; k++) {
                            double dist = (_centroids.row(k) - data.row(i)).squaredNorm();
                            if (dist < min) {
                                min = dist;
                                min_k = k;
                                _labels(i) = min_k;
                                _inertia += dist;
                            }
                        }

                        _clusters[min_k].conservativeResize(_clusters[min_k].rows() + 1,
                                                            data.cols());
                        _clusters[min_k].row(_clusters[min_k].rows() - 1) = data.row(i);
                    }
                    _inertia /= data.rows();

                    if (prev_centroids.size() && (prev_centroids == _centroids))
                        break; // algorithm has converged
                    else
                        prev_centroids = _centroids;

                    // update centroids
                    for (int k = 0; k < K; ++k) {
                        if (_clusters[k].size() == 0)
                            _centroids.row(k) = Eigen::VectorXd::Zero(data.cols());
                        else
                            _centroids.row(k) = _clusters[k].colwise().mean();
                    }
                }

                return _clusters;
            }

            const Eigen::MatrixXd& centroids() const { return _centroids; }
            double inertia() const { return _inertia; }
            const Eigen::VectorXi& labels() const { return _labels; }

          protected:
            Eigen::MatrixXd _centroids;
            std::vector<Eigen::MatrixXd> _clusters;
            Eigen::VectorXi _labels;
            double _inertia;
        };
    } // namespace clustering
} // namespace aegean

#endif