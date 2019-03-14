#ifndef AEGEAN_FEATURES_ANGULAR_DISPLACEMENT_VELOCITY_HPP
#define AEGEAN_FEATURES_ANGULAR_DISPLACEMENT_VELOCITY_HPP

#include <cmath>
#include <features/angular_position.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        template <typename CircularPolygon>
        class AngularDisplacementVelocity : public AngularPosition<CircularPolygon> {
            using base_t = AngularPosition<CircularPolygon>;

        public:
            AngularDisplacementVelocity() : _feature_name("angular_displacement_velocity") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                using namespace tools;
                base_t::operator()(matrix, timestep);
                Eigen::MatrixXd angular_pos = base_t::get();

                const uint duration = matrix.rows();
                _angular_displacement_velocity = Eigen::MatrixXd::Zero(duration, angular_pos.cols());
                for (uint i = 1; i < angular_pos.rows(); ++i) {
                    for (uint j = 0; j < angular_pos.cols(); ++j) {
                        double phi = -(angular_pos(i, j) - angular_pos(i - 1, j));
                        if (std::abs(phi) > 180.)
                            phi = -sgn(phi) * (360. - std::abs(phi));
                        _angular_displacement_velocity(i, j) = phi / timestep;
                    }
                }

                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(_angular_displacement_velocity.cols()) * 3;
                for (uint i = 0; i < _angular_displacement_velocity.cols(); ++i)
                    _angular_displacement_velocity(0, i) = _angular_displacement_velocity(1, i) + noise(i);
            }

            Eigen::MatrixXd get() override { return _angular_displacement_velocity; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _angular_displacement_velocity;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif