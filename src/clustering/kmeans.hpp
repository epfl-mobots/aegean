#ifndef AEGEAN_CLUSTERING_KMEANS_HPP
#define AEGEAN_CLUSTERING_KMEANS_HPP

#include <map>
#include <vector>

#include <Eigen/Core>

// Quick hack for definition of 'I' in <complex.h>
#undef I

#include <limbo/tools/random_generator.hpp>

namespace aegean {
    namespace clustering {
        /// Cluster the data (NxD) in K clusters
        /// returns a vector with the clusters (NxD)
        std::vector<Eigen::MatrixXd> kmeans(const Eigen::MatrixXd& data, int K, int max_iter = 100)
        {
            static thread_local tools::rgen_int_t rgen(0, data.rows() - 1);

            Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(K, data.cols());
            // random points as centroids in the beginning
            for (int i = 0; i < K; i++) {
                centroids.row(i) = data.row(rgen.rand());
            }

            std::vector<Eigen::MatrixXd> clusters;

            for (int n = 0; n < max_iter; n++) {
                // assign points to cluster
                clusters.clear();
                clusters.resize(K);

                for (int i = 0; i < data.rows(); i++) {
                    double min = std::numeric_limits<double>::max();
                    int min_k = -1;
                    for (int k = 0; k < K; k++) {
                        double dist = (centroids.row(k) - data.row(i)).squaredNorm();
                        if (dist < min) {
                            min = dist;
                            min_k = k;
                        }
                    }

                    clusters[min_k].conservativeResize(clusters[min_k].rows() + 1, data.cols());
                    clusters[min_k].row(clusters[min_k].rows() - 1) = data.row(i);
                }

                // update centroids
                for (int k = 0; k < K; k++) {
                    if (clusters[k].size() == 0)
                        centroids.row(k) = Eigen::VectorXd::Zero(data.cols());
                    else
                        centroids.row(k) = clusters[k].colwise().mean();
                }
            }

            return clusters;
        }
    } // namespace clustering
} // namespace aegean

#endif