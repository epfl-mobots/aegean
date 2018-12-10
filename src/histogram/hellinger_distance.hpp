#ifndef AEGEAN_HISTOGRAM_HELLINGER_DISTANCE_HPP
#define AEGEAN_HISTOGRAM_HELLINGER_DISTANCE_HPP

#include <Eigen/Core>
#include <cassert>

namespace aegean {
    namespace histogram {

        class HellingerDistance {
        public:
            HellingerDistance() {}

            double operator()(const Eigen::MatrixXd& hist1, const Eigen::MatrixXd& hist2) const
            {
                assert((hist1.rows() == hist2.rows()) && (hist1.cols() == hist2.cols()) && "Dimensions don't match");
                assert(hist1.rows() == 1);
                double h = std::sqrt(0.5 * ((hist1.array().sqrt() - hist2.array().sqrt()).pow(2).sum()));
                return h;
            }
        };

    } // namespace histogram
} // namespace aegean

#endif