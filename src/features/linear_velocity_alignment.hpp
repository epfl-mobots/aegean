#ifndef AEGEAN_FEATURES_LINEAR_VELOCITY_ALIGNMENT_HPP
#define AEGEAN_FEATURES_LINEAR_VELOCITY_ALIGNMENT_HPP

#include <cmath>
#include <features/bearing.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class Alignment : public Bearing {
        public:
            Alignment(bool in_deg = false) : _feature_name("linear_velocity_alignment"), _in_deg(in_deg) {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                const uint duration = matrix.rows();

                Eigen::VectorXd noise = Eigen::VectorXd::Ones(matrix.cols())
                    - limbo::tools::random_vector_bounded(matrix.cols()) / 10;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, -1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(duration - 1, i) = matrix(duration - 1, i) * noise(i);
                Eigen::MatrixXd velocities = (rolled - matrix) / timestep;
            }

            Eigen::MatrixXd get() override { return _lin_vel_align; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _lin_vel_align;

        private:
            std::string _feature_name;
            bool _in_deg;
        };

    } // namespace features
} // namespace aegean

#endif