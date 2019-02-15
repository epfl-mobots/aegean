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
                using namespace tools;

                Bearing::operator()(matrix, timestep);
                Eigen::MatrixXd bearings = Bearing::get();

                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(bearings.cols()) * 3;
                Eigen::MatrixXd rolled = tools::rollMatrix(bearings, 1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(0, i) = bearings(0, i) + noise(i);

                _angular_velocity = -(bearings - rolled);
                _angular_velocity.unaryExpr([timestep](double val) {
                    double corrected_phi = val;
                    if (abs(val) > 180)
                        corrected_phi = sgn(corrected_phi) * (360 - corrected_phi);
                    return corrected_phi / timestep;
                }); // result in degrees
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