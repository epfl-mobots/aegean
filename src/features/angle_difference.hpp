#ifndef AEGEAN_FEATURES_ANGLE_DIFFERENCE_HPP
#define AEGEAN_FEATURES_ANGLE_DIFFERENCE_HPP

#include <cmath>
#include <features/alignment.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class AngleDifference : public Bearing {
        public:
            AngleDifference() : _feature_name("angle_difference") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                using namespace tools;

                uint duration = matrix.rows();
                Bearing::operator()(matrix, timestep);

                _diff.resize(matrix.cols() / 2, Eigen::MatrixXd::Zero(duration, matrix.cols() / 2));
                for (uint r = 0; r < duration; ++r) {
                    for (uint k = 0; k < matrix.cols() / 2; ++k) {
                        for (uint l = 0; l < matrix.cols() / 2; ++l) {
                            if (k == l) // k is the focal individual
                                continue;
                            double dbearing = -(_bearing(r, k) - _bearing(r, l)); // we want CW to be negative and CCW positive (just a convention)
                            if (abs(dbearing) > 180)
                                dbearing = -1 * sgn(dbearing) * (360 - abs(dbearing));
                            _diff[k](r, l) = dbearing;
                        } // l
                    } // k
                } // r
            }

            virtual const std::vector<Eigen::MatrixXd>& get_vec() { return _diff; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            std::vector<Eigen::MatrixXd> _diff;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif