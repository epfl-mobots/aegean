#ifndef LIMBO_MODEL_CLUSTERING_CLUSTERDP_HPP
#define LIMBO_MODEL_CLUSTERING_CLUSTERDP_HPP

#include <tools/archive.hpp>

#include <map>
#include <vector>
#include <Eigen/Core>
#include <algorithm>
#include <numeric>

#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/parameter.hpp>

#include <tools/random_generator.hpp>

namespace aegean {
    namespace defaults {
        struct Clusterdp {
            static constexpr double dc = -1;
            static constexpr double p = .02;
        };

        struct Chi {
            double operator()(double distance, double cutoff) const
            {
                return (distance < cutoff) ? 1 : 0;
            }
        };

        struct Exp {
            double operator()(double distance, double cutoff) const
            {
                return std::exp(-std::pow(distance / cutoff, 2.));
            }
        };

        struct Euclidean {
            double operator()(Eigen::VectorXd pt1, Eigen::VectorXd pt2) const
            {
                return (pt1 - pt2).norm();
            }
        };
    } // namespace defaults

    namespace clustering {

        BOOST_PARAMETER_TEMPLATE_KEYWORD(densityfun)
        BOOST_PARAMETER_TEMPLATE_KEYWORD(distancefun)

        using clusterdp_signature
            = boost::parameter::parameters<
                boost::parameter::optional<tag::densityfun>,
                boost::parameter::optional<tag::distancefun>>;

        template <typename Params, class A1 = boost::parameter::void_, class A2 = boost::parameter::void_>
        class Clusterdp {

            struct defaults {
                using distance_t = aegean::defaults::Euclidean;
                using density_t = aegean::defaults::Chi;
            };

            using args = typename clusterdp_signature::bind<A1, A2>::type;
            using DensityFunc =
                typename boost::parameter::binding<args, tag::densityfun, typename defaults::density_t>::type;
            using DistanceFunc =
                typename boost::parameter::binding<args, tag::distancefun, typename defaults::distance_t>::type;

        public:
            Clusterdp() {}

            std::vector<Eigen::MatrixXd> fit(const Eigen::MatrixXd& data, const int K)
            {
                // point distances
                _compute_distances(data);

                // initialiaze the distance cutoff value either by a user value
                // or by estimating the value
                if (Params::Clusterdp::dc >= 0) {
                    _dc = Params::Clusterdp::dc;
                }
                else {
                    int N = _distances.rows();
                    Eigen::VectorXd flat_distances((N * N - N) / 2);
                    int cur_idx = 0;
                    for (uint i = 0; i < data.rows(); ++i) {
                        for (uint j = i + 1; j < data.rows(); ++j) {
                            flat_distances(cur_idx++) = _distances(i, j);
                        }
                    }
                    std::vector<size_t> idx(flat_distances.rows());
                    std::iota(idx.begin(), idx.end(), 0);
                    std::sort(idx.begin(), idx.end(), [&flat_distances](const size_t lhs, const size_t rhs) {
                        return flat_distances(lhs) < flat_distances(rhs);
                    });
                    int position = (N * N - 1) * Params::Clusterdp::p;
                    _dc = flat_distances(idx[position]);
                }

                // compute point densities
                _compute_rhos();

                // compute point minimum distances from denser points
                Eigen::VectorXi nn_idcs = _compute_deltas();

                // sort the corresponding density indices in descending order
                std::vector<size_t> idx(_deltas.rows());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](const size_t lhs, const size_t rhs) {
                    return _deltas(lhs) > _deltas(rhs);
                });

                // the centroids are the points with anomalously large delta
                _labels = Eigen::VectorXi::Ones(_deltas.rows()) * -1;
                _centroids = Eigen::MatrixXd::Zero(K, data.cols());
                for (int i = 0; i < K; ++i) {
                    _centroids.row(i) = data.row(idx[i]);
                    _labels(idx[i]) = i;
                }

                // clustering starts from the points with highest density and propagates
                // to the nearest neighbours
                idx.resize(_rhos.rows());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](const size_t lhs, const size_t rhs) {
                    return _rhos(lhs) > _rhos(rhs);
                });

                for (uint i = 0; i < data.rows(); ++i) {
                    if (_labels(idx[i]) < 0)
                        _labels(idx[i]) = _labels(nn_idcs(idx[i]));
                }

                // computing border density conditions for the halo points
                Eigen::VectorXd border_rho = Eigen::VectorXd::Zero(_rhos.rows());
                for (int i = 0; i < data.rows() - 1; ++i) {
                    for (int j = i + 1; j < data.rows(); ++j) {
                        if (_labels(i) != _labels(j) && _distances(i, j) <= _dc) {
                            double avg_rho = (_rhos(i) + _rhos(j)) / 2;
                            if (avg_rho > border_rho(_labels(i)))
                                border_rho(_labels(i)) = avg_rho;
                            if (avg_rho > border_rho(_labels(j)))
                                border_rho(_labels(j)) = avg_rho;
                        }
                    }
                }

                // mark halo ponits
                _halo_flags = Eigen::VectorXi::Zero(_labels.rows());
                for (int i = 0; i < data.rows(); ++i) {
                    if (_rhos(i) < border_rho(_labels(i)))
                        _halo_flags(i) = 1;
                }

                // assign points to corresponding clusters or to halos
                _clusters.clear();
                _clusters.resize(K);
                for (uint i = 0; i < data.rows(); ++i) {
                    if (_halo_flags(i)) {
                        _halos.conservativeResize(_halos.rows() + 1, data.cols());
                        _halos.row(_halos.rows() - 1) = data.row(i);
                    }
                    else {
                        _clusters[_labels(i)].conservativeResize(_clusters[_labels(i)].rows() + 1, data.cols());
                        _clusters[_labels(i)].row(_clusters[_labels(i)].rows() - 1) = data.row(i);
                    }
                }

                return _clusters;
            }

            Eigen::MatrixXi predict(const Eigen::MatrixXd& data) const
            {
                assert(data.cols() == _centroids.cols());
                Eigen::MatrixXi predictions(data.rows(), 1);
                for (int i = 0; i < data.rows(); ++i) {
                    double min_d = std::numeric_limits<double>::max();
                    int min_k = -1;
                    for (int k = 0; k < _centroids.rows(); ++k) {
                        double dist = _df(_centroids.row(k), data.row(i));
                        if (dist < min_d) {
                            min_d = dist;
                            min_k = k;
                        }
                    }
                    predictions(i) = min_k;
                }
                return predictions;
            }

            void save(const tools::Archive& archive) const
            {
                archive.save(_centroids, "centroids_clusterdp");
                archive.save(_labels, "labels_clusterdp");
                archive.save(_halos, "halos_clusterdp");
                for (uint i = 0; i < _clusters.size(); ++i)
                    archive.save(_clusters[i], "cluster_" + std::to_string(i) + "_data");
            }

            const Eigen::MatrixXd& centroids() const { return _centroids; }
            const Eigen::MatrixXd& halos() const { return _halos; }
            const Eigen::VectorXi& labels() const { return _labels; }

        protected:
            void _compute_distances(const Eigen::MatrixXd& data)
            {
                // notice that we assume that the samples are stored by row
                _distances = Eigen::MatrixXd::Zero(data.rows(), data.rows());
                for (uint i = 0; i < data.rows(); ++i) {
                    for (uint j = i + 1; j < data.rows(); ++j) { // only traversing half of the symmetric matrix
                        _distances(i, j) = _df(data.row(i), data.row(j));
                        _distances(j, i) = _distances(i, j);
                    }
                }
            }

            void _compute_rhos()
            {
                _rhos = Eigen::VectorXd::Zero(_distances.rows());
                for (uint i = 0; i < _distances.rows(); ++i) {
                    for (uint j = i + 1; j < _distances.cols(); ++j) {
                        _rhos(i) += _chi(_distances(i, j), _dc);
                        _rhos(j) += _chi(_distances(i, j), _dc);
                    }
                }
            }

            Eigen::VectorXi _compute_deltas()
            {
                Eigen::VectorXi nn_idcs = Eigen::VectorXi::Zero(_distances.rows());
                _deltas = Eigen::VectorXd::Zero(_distances.rows());
                for (uint i = 0; i < _distances.rows(); ++i) {
                    std::vector<std::pair<double, uint>> higher_density_distances;
                    for (uint j = 0; j < _distances.cols(); ++j) {
                        if (_rhos(i) < _rhos(j)) {
                            higher_density_distances.push_back(std::make_pair(_distances(i, j), j));
                        }
                    }

                    if (higher_density_distances.size() == 0) {
                        Eigen::VectorXd::Index max_ridx, max_cidx;
                        _deltas(i) = _distances.maxCoeff(&max_ridx, &max_cidx);
                    }
                    else {
                        std::vector<size_t> idx(higher_density_distances.size());
                        std::iota(idx.begin(), idx.end(), 0);
                        std::sort(idx.begin(), idx.end(), [&higher_density_distances](const size_t lhs, const size_t rhs) {
                            return higher_density_distances[lhs].first < higher_density_distances[rhs].first;
                        });
                        _deltas(i) = higher_density_distances[idx[0]].first;
                        nn_idcs(i) = higher_density_distances[idx[0]].second;
                    }
                }
                return nn_idcs;
            }

            Eigen::MatrixXd _centroids;
            std::vector<Eigen::MatrixXd> _clusters;
            Eigen::MatrixXd _distances;
            Eigen::VectorXd _rhos;
            Eigen::VectorXd _deltas;
            Eigen::VectorXi _labels;
            double _dc;

            Eigen::MatrixXd _halos;
            Eigen::VectorXi _halo_flags;

            DistanceFunc _df;
            DensityFunc _chi;
        };

    } // namespace clustering
} // namespace aegean

#endif