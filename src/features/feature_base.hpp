#ifndef AEGEAN_FEATURES_FEATURE_BASE_HPP
#define AEGEAN_FEATURES_FEATURE_BASE_HPP

#include <Eigen/Core>
#include <string>

namespace aegean {
    namespace features {

        class FeatureBase {
        public:
            virtual void operator()(const Eigen::MatrixXd& matrix, const float timestep)
            {
                assert(false);
            }
            virtual Eigen::MatrixXd get() { assert(false); }
            virtual const std::vector<Eigen::MatrixXd>& get_vec() { assert(false); }
            virtual const std::string& feature_name() { assert(false); }
        };

    } // namespace features
} // namespace aegean

#endif