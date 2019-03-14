#ifndef AEGEAN_FEATURES_ANGULAR_POSITION_HPP
#define AEGEAN_FEATURES_ANGULAR_POSITION_HPP

#include <cmath>
#include <features/feature_base.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        template <typename CircularPolygon>
        class AngularPosition : public FeatureBase {
        public:
            AngularPosition() : _feature_name("angular_position") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                using namespace tools;
                using namespace primitives;

                const uint duration = matrix.rows();

                _angular_position = Eigen::MatrixXd::Zero(duration, matrix.cols() / 2);
                for (uint i = 0; i < _angular_position.rows(); ++i) {
                    for (uint j = 0; j < _angular_position.cols(); ++j) {
                        Point p;
                        p.x() = matrix(i, j * 2);
                        p.y() = matrix(i, j * 2 + 1);
                        _angular_position(i, j) = _cp.angle(p);
                    }
                }
            }

            Eigen::MatrixXd get() override { return _angular_position; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _angular_position;

        private:
            CircularPolygon _cp;
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif