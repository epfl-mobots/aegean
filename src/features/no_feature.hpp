#ifndef NO_FEATURE_HPP
#define NO_FEATURE_HPP

#include <features/feature_base.hpp>

namespace aegean {
    namespace features {

        class NoFeature : public FeatureBase {
          public:
            NoFeature() : _feature_name("no_feature") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override {}

            Eigen::MatrixXd get() override { return Eigen::MatrixXd(); }

            const std::string& feature_name() override { return _feature_name; }

          private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif