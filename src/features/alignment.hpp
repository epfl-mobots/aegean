#ifndef AEGEAN_FEATURES_ALIGNMENT_HPP
#define AEGEAN_FEATURES_ALIGNMENT_HPP

#include <cmath>
#include <features/bearing.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

namespace aegean {
    namespace features {

        class Alignment : public Bearing {
        public:
            Alignment() : _feature_name("alignment") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                Bearing::operator()(matrix, timestep);
                _alignment = Eigen::MatrixXd::Zero(matrix.rows(), 1);
                for (uint i = 0; i < matrix.rows(); ++i) {
                    for (uint j = 0; j < matrix.cols() / 2; ++j) {
                        double ind_avg = _bearing.row(i)
                                             .unaryExpr([&](double val) {
                                                 double theta = fabs(val - _bearing(i, j));
                                                 return (theta > 180) ? 360 - theta : theta;
                                             })
                                             .sum()
                            / (_bearing.cols() - 1);
                        _alignment(i) += ind_avg;
                    }
                    _alignment(i) /= _bearing.cols();
                }
                _alignment = (_alignment / 360).unaryExpr([](double val) { return 1 - val; });
            }

            Eigen::MatrixXd get() override { return _alignment; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            Eigen::MatrixXd _alignment;

        private:
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif