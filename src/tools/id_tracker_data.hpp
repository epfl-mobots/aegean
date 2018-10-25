#ifndef ID_TRACKER_DATA_HPP
#define ID_TRACKER_DATA_HPP

#include <Eigen/Core>
#include <tools/reconstruction/no_reconstruction.hpp>

namespace aegean {
    namespace tools {

        template <typename ReconstructionMethod = reconstruction::NoReconstruction>
        class IdTrackerData {

          public:
            IdTrackerData(const Eigen::MatrixXd& positions, const uint fps = 15,
                          const uint centroid_samples = 0, const float scale = 1.0f,
                          const uint skip_rows = 0)
                : _positions(positions),
                  _fps(fps),
                  _centroid_samples(centroid_samples),
                  _scale(scale),
                  _skip_rows(skip_rows),
                  _num_individuals(_positions.cols() / 2)
            {
                _positions *= scale;
                if (_centroid_samples > 1)
                    _filter_positions();
                _reconstruction_method(_positions);
            }

            const Eigen::MatrixXd& positions() const { return _positions; }
            uint fps() const { return _fps; }
            uint centroid_samples() const { return _centroid_samples; }
            float scale() const { return _scale; }
            uint skip_rows() const { return _skip_rows; }

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
        };
    } // namespace tools
} // namespace aegean

#endif