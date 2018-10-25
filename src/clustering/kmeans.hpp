#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <cassert>
#include <Eigen/Core>

namespace aegean {
    namespace clustering {
        class KMeans {
          public:
            KMeans() {}

            void fit(const Eigen::MatrixXd& data, int K, int max_iter = 100) {}

            void predict(const Eigen::MatrixXd& data) {}

          protected:
            Eigen::MatrixXd _data;
            Eigen::MatrixXd _centers;
            double _inertia;
        };
    } // namespace clustering
} // namespace aegean

#endif