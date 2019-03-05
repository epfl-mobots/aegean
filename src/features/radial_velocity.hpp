#ifndef AEGEAN_FEATURES_RADIAL_VELOCITY_HPP
#define AEGEAN_FEATURES_RADIAL_VELOCITY_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        template <typename CircularPolygon>
        class RadialVelocity : public FeatureBase {
        public:
            RadialVelocity() : _feature_name("radial_velocity") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                using namespace tools;
                using namespace primitives;

                const uint duration = matrix.rows();

                Eigen::MatrixXd r(duration, matrix.cols() / 2);
                for (uint i = 0; i < r.rows(); ++i) {
                    for (uint j = 0; j < r.cols(); ++j) {
                        Point p;
                        p.x() = matrix(i, j * 2);
                        p.y() = matrix(i, j * 2 + 1);
                        r(i, j) = _cp.distance_to_center(p);
                    }
                }

                Eigen::VectorXd noise = Eigen::VectorXd::Ones(r.cols())
                    - limbo::tools::random_vector_bounded(r.cols()) * 0.01;
                Eigen::MatrixXd rolled = tools::rollMatrix(r, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = r(0, i) + noise(i);

                _radial_velocity = (r - rolled) / timestep;
            }

            Eigen::MatrixXd get() override { return _radial_velocity; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _radial_velocity;

        private:
            CircularPolygon _cp;
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif