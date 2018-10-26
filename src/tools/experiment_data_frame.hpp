#ifndef EXPERIMENT_DATA_FRAME_HPP
#define EXPERIMENT_DATA_FRAME_HPP

#include <Eigen/Core>
#include <tools/reconstruction/no_reconstruction.hpp>
#include <features/no_feature.hpp>

#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/parameter.hpp>

namespace aegean {

    struct ConstructFeatures {
        ConstructFeatures(const Eigen::MatrixXd& trajectories) : _trajectories(trajectories) {}

        template <typename T>
        void operator()(T& x)
        {
            if (x.feature_name() != "no_feature") {
                x(_trajectories);
                _results.push_back(x.get());
            }
        }

        std::vector<Eigen::MatrixXd> results() { return _results; }

        Eigen::MatrixXd _trajectories;
        std::vector<Eigen::MatrixXd> _results;
    };

    namespace tools {

        BOOST_PARAMETER_TEMPLATE_KEYWORD(reconfun)
        BOOST_PARAMETER_TEMPLATE_KEYWORD(features)

        using edf_signature
            = boost::parameter::parameters<boost::parameter::optional<tag::reconfun>,
                                           boost::parameter::optional<tag::features>>;

        template <class A1 = boost::parameter::void_, class A2 = boost::parameter::void_>
        class ExperimentDataFrame {

            struct defaults {
                using reconstruction_t = reconstruction::NoReconstruction;
                using features_t = boost::fusion::vector<aegean::features::NoFeature>;
            };

            using args = typename edf_signature::bind<A1, A2>::type;
            using ReconstructionMethod =
                typename boost::parameter::binding<args, tag::reconfun,
                                                   typename defaults::reconstruction_t>::type;
            using Features =
                typename boost::parameter::binding<args, tag::features,
                                                   typename defaults::features_t>::type;

          public:
            ExperimentDataFrame(const Eigen::MatrixXd& positions, const uint fps = 15,
                                const uint centroid_samples = 0, const float scale = 1.0f,
                                const uint skip_rows = 0)
                : _positions(positions),
                  _fps(fps),
                  _centroid_samples(centroid_samples),
                  _scale(scale),
                  _skip_rows(skip_rows),
                  _num_individuals(_positions.cols() / 2)
            {
                // scale position matrix (e.g., this can be used to convert from pixels to cm)
                _positions *= scale;

                // filter positions by calculating centroidal positions if necessary
                if (_centroid_samples > 1)
                    _filter_positions();

                // reconstruct missing (NaN) values
                _reconstruction_method(_positions);

                // compute the features
                ConstructFeatures cf(_positions);
                boost::fusion::for_each(_features, cf);
                _feature_res = cf.results();
            }

            const Eigen::MatrixXd& positions() const { return _positions; }
            uint fps() const { return _fps; }
            uint centroid_samples() const { return _centroid_samples; }
            float scale() const { return _scale; }
            uint skip_rows() const { return _skip_rows; }
            uint num_individuals() const { return _num_individuals; }

          protected:
            void _filter_positions()
            {
                Eigen::MatrixXd centroidal;
                for (uint i = 0; i < _positions.rows(); i += _centroid_samples) {
                    Eigen::MatrixXd block
                        = _positions.block(i, 0, _centroid_samples, _num_individuals * 2);

                    Eigen::MatrixXd nans = block.array().isNaN().select(block, 0);
                    Eigen::MatrixXd denom
                        = nans.array()
                              .isNaN()
                              .select(0, Eigen::MatrixXd::Ones(block.rows(), block.cols()))
                              .colwise()
                              .sum();
                    Eigen::MatrixXd centroidal_position
                        = block.array().isNaN().select(0, block).colwise().sum();

                    for (uint j = 0; j < centroidal_position.cols(); ++j) {
                        if (denom(j) > 0)
                            centroidal_position(j) /= denom(j);
                        else
                            centroidal_position(j) = std::nan("NaN");
                    }
                    centroidal.conservativeResize(centroidal.rows() + 1, _num_individuals * 2);
                    centroidal.row(centroidal.rows() - 1) = centroidal_position;
                }
                _positions = centroidal;
            }

            Eigen::MatrixXd _positions;
            const uint _fps;
            const uint _centroid_samples;
            const float _scale;
            const uint _skip_rows;
            const uint _num_individuals;

            ReconstructionMethod _reconstruction_method;
            Features _features;
            std::vector<Eigen::MatrixXd> _feature_res;
        };
    } // namespace tools
} // namespace aegean

#endif