#ifndef AEGEAN_FEATURES_ANGULAR_VELOCITY_HPP
#define AEGEAN_FEATURES_ANGULAR_VELOCITY_HPP

#include <cmath>
#include <features/bearing.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class AngularVelocity : public Bearing {
        public:
            AngularVelocity() : _feature_name("angular_velocity") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                Bearing::operator()(matrix, timestep);
                Eigen::MatrixXd bearings = Bearing::get();

                Eigen::VectorXd noise = Eigen::VectorXd::Ones(bearings.cols())
                    - limbo::tools::random_vector_bounded(bearings.cols()) * 5;
                Eigen::MatrixXd rolled = tools::rollMatrix(bearings, -1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(duration - 1, i) = bearings(duration - 1, i) * noise(i);
                _angular_velocity = (rolled - bearings);
                _angular_velocity.unaryExpr([timestep](double val) { return std::fmod(val, 360.) / timestep; }); // result in degrees
            }

            Eigen::MatrixXd get() override { return _angular_velocity; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _angular_velocity;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif