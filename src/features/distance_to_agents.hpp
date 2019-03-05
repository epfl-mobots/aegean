#ifndef AEGEAN_FEATURES_DISTANCE_TO_AGENTS_HPP
#define AEGEAN_FEATURES_DISTANCE_TO_AGENTS_HPP

#include <cmath>
#include <features/inter_individual_distance.hpp>
#include <tools/mathtools.hpp>
#include <tools/random_generator.hpp>

#include <iostream>

namespace aegean {
    namespace features {

        template <typename DistanceFunc>
        class DistanceToAgents : public FeatureBase {
        public:
            DistanceToAgents() : _feature_name("distance_to_agents") {}

            void operator()(const Eigen::MatrixXd& matrix, const float timestep) override
            {
                uint duration = matrix.rows();

                _distances.resize(matrix.cols() / 2, Eigen::MatrixXd::Zero(duration, matrix.cols() / 2));
                for (uint r = 0; r < matrix.rows(); ++r) {
                    for (uint k = 0; k < matrix.cols() / 2; ++k) {
                        for (uint l = 0; l < matrix.cols() / 2; ++l) {
                            if (l == k)
                                continue; // skipping distance of individual withitself
                            Eigen::MatrixXd focal(2, 1), neigh(2, 1);
                            focal(0) = matrix(r, k * 2);
                            focal(1) = matrix(r, k * 2 + 1);
                            neigh(0) = matrix(r, l * 2);
                            neigh(1) = matrix(r, l * 2 + 1);
                            _distances[k](r, l) = _distance(focal, neigh); // every column k contains the distances of the kth individual to the neighbours
                        } // l
                    } // k
                } // r
            }

            Eigen::MatrixXd get() override { assert(false); };
            virtual const std::vector<Eigen::MatrixXd>& get_vec() { return _distances; }

            const std::string& feature_name() override { return _feature_name; }

        protected:
            std::vector<Eigen::MatrixXd> _distances;

        private:
            DistanceFunc _distance;
            std::string _feature_name;
        };

    } // namespace features
} // namespace aegean

#endif