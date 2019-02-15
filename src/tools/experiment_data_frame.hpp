#ifndef AEGEAN_TOOLS_EXPERIMENT_DATA_FRAME_HPP
#define AEGEAN_TOOLS_EXPERIMENT_DATA_FRAME_HPP

#include <Eigen/Core>
#include <features/no_feature.hpp>
#include <tools/reconstruction/no_reconstruction.hpp>
#include <tools/mathtools.hpp>
#include <limbo/tools/random_generator.hpp>

#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/parameter.hpp>

#include <memory>

namespace aegean {

    using ResultVecPtr = std::shared_ptr<std::vector<Eigen::MatrixXd>>;
    using NameVecPtr = std::shared_ptr<std::vector<std::string>>;

    struct ConstructFeatures {
        ConstructFeatures(const Eigen::MatrixXd& trajectories, ResultVecPtr results,
            const float timestep, const uint skip_rows)
            : _trajectories(trajectories),
              _results(results),
              _timestep(timestep),
              _skip_rows(skip_rows)
        {
        }

        template <typename T>
        void operator()(T& x)
        {
            if (x.feature_name() != "no_feature") {
                x(_trajectories, _timestep);
                _results->push_back(
                    x.get().block(_skip_rows, 0, x.get().rows() - _skip_rows, x.get().cols()));
            }
        }

        Eigen::MatrixXd _trajectories;
        ResultVecPtr _results;
        const float _timestep;
        const uint _skip_rows;
    };

    struct FeatureNames {
        FeatureNames(NameVecPtr names) : _names(names) {}

        template <typename T>
        void operator()(T& x)
        {
            _names->push_back(x.feature_name());
        }

        NameVecPtr _names;
    };

    namespace tools {

        BOOST_PARAMETER_TEMPLATE_KEYWORD(reconfun)
        BOOST_PARAMETER_TEMPLATE_KEYWORD(featset)

        using edf_signature
            = boost::parameter::parameters<
                boost::parameter::optional<tag::reconfun>,
                boost::parameter::optional<tag::featset>>;

        template <class A1 = boost::parameter::void_, class A2 = boost::parameter::void_>
        class ExperimentDataFrame {

            struct defaults {
                using reconstruction_t = reconstruction::NoReconstruction;
                using features_t = boost::fusion::vector<aegean::features::NoFeature>;
            };

            using args = typename edf_signature::bind<A1, A2>::type;
            using ReconstructionMethod =
                typename boost::parameter::binding<args, tag::reconfun, typename defaults::reconstruction_t>::type;
            using Features =
                typename boost::parameter::binding<args, tag::featset, typename defaults::features_t>::type;

        public:
            ExperimentDataFrame(const Eigen::MatrixXd& positions, const uint fps = 15,
                const uint centroid_samples = 0, const float scale = 1.0f,
                const uint skip_rows = 0)
                : _positions(positions),
                  _fps(fps),
                  _centroid_samples(centroid_samples),
                  _timestep(static_cast<float>(_centroid_samples) / static_cast<float>(_fps)),
                  _scale(scale),
                  _skip_rows(skip_rows / _centroid_samples),
                  _num_individuals(_positions.cols() / 2),
                  _feature_res(std::make_shared<std::vector<Eigen::MatrixXd>>()),
                  _feature_names(std::make_shared<std::vector<std::string>>())
            {
                // scale position matrix (e.g., this can be used to convert from pixels to cm)
                _positions *= scale;

                // filter positions by calculating centroidal positions if necessary
                if (_centroid_samples > 1)
                    _filter_positions();

                // reconstruct missing (NaN) values
                _reconstruction_method(_positions);

                // compute the features
                ConstructFeatures cf(_positions, _feature_res, _timestep, _skip_rows);
                boost::fusion::for_each(_features, cf);

                FeatureNames fn(_feature_names);
                boost::fusion::for_each(_features, fn);

                // use the extra rows to compute the features and then remove them
                _compute_velocities();

                // eigen does not always behave well with matrix = matrix.block(...)
                {
                    Eigen::MatrixXd bl = _positions.block(_skip_rows, 0, _positions.rows() - _skip_rows, _positions.cols());
                    _positions = bl;
                }
                {
                    Eigen::MatrixXd bl = _velocities.block(_skip_rows, 0, _velocities.rows() - _skip_rows, _velocities.cols());
                    _velocities = bl;
                }
            }

            const Eigen::MatrixXd& positions() const { return _positions; }
            const Eigen::MatrixXd& velocities() const { return _velocities; }
            uint fps() const { return _fps; }
            uint centroid_samples() const { return _centroid_samples; }
            float scale() const { return _scale; }
            uint skip_rows() const { return _skip_rows; }
            uint num_individuals() const { return _num_individuals; }

            Eigen::MatrixXd get_feature_matrix() const
            {
                if (_feature_res->size() == 0)
                    return Eigen::MatrixXd();

                Eigen::MatrixXd feature_mat((*_feature_res)[0].rows(), _feature_res->size());
                for (uint i = 0; i < _feature_res->size(); ++i)
                    feature_mat.col(i) = (*_feature_res)[i];
                return feature_mat;
            }

            std::vector<Eigen::MatrixXd> features() const { return *_feature_res; }
            std::vector<std::string> feature_names() const
            {
                return *_feature_names;
            }

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
                        if (denom(j) > 0) {
                            centroidal_position(j) /= denom(j);

                            if (denom(j) < 1)
                                std::cout << denom(j) << std::endl;
                        }
                        else
                            centroidal_position(j) = std::nan("NaN");
                    }
                    centroidal.conservativeResize(centroidal.rows() + 1, _num_individuals * 2);
                    centroidal.row(centroidal.rows() - 1) = centroidal_position;
                }
                _positions = centroidal;
            }

            void _compute_velocities()
            {
                _velocities = _compute_diff(_positions);
            }

            Eigen::MatrixXd _compute_diff(const Eigen::MatrixXd& matrix) const
            {
                Eigen::MatrixXd diff;
                const uint duration = matrix.rows();
                Eigen::VectorXd noise = limbo::tools::random_vector_bounded(matrix.cols()) * 0.01;
                Eigen::MatrixXd rolled = tools::rollMatrix(matrix, -1);
                for (uint i = 0; i < rolled.cols(); ++i)
                    rolled(duration - 1, i) = matrix(duration - 1, i) + noise(i);
                diff = (rolled - matrix) / _timestep;
                return diff;
            }

            Eigen::MatrixXd _positions;
            Eigen::MatrixXd _velocities;
            const uint _fps;
            const uint _centroid_samples;
            const float _timestep;
            const float _scale;
            const uint _skip_rows;
            const uint _num_individuals;

            ReconstructionMethod _reconstruction_method;
            Features _features;
            ResultVecPtr _feature_res;
            NameVecPtr _feature_names;
        };
    } // namespace tools
} // namespace aegean

#endif