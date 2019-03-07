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
            int operator()(double distance) const
            {
                return (distance < 0) ? 1 : 0;
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

            struct Individual {
                Individual() : cluster_id(-1), idx(-1), nn_idx(-1) {}
                int cluster_id;
                int idx;
                int nn_idx;
            };

            using IndividualPtr = std::shared_ptr<Individual>;

        public:
            Clusterdp()
            {
            }

            std::vector<Eigen::MatrixXd> fit(const Eigen::MatrixXd& data, const int K)
            {
                Eigen::MatrixXd distances = _compute_distances(data);

                if (Params::Clusterdp::dc >= 0) {
                    _dc = Params::Clusterdp::dc;
                }
                else {
                    Eigen::VectorXd flat_distances(
                        Eigen::Map<Eigen::VectorXd>(distances.data(), distances.cols() * distances.rows()));
                    std::vector<size_t> idx(flat_distances.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    std::sort(idx.begin(), idx.end(), [&flat_distances](const size_t lhs, const size_t rhs) {
                        return flat_distances(lhs) < flat_distances(rhs);
                    });
                    size_t start_idx;
                    for (uint i = 0; i < idx.size(); ++i) {
                        if (flat_distances(idx[i]) > 0) {
                            start_idx = i;
                            break;
                        }
                    }
                    int position = flat_distances.rows() * Params::Clusterdp::p + start_idx;
                    _dc = flat_distances(idx[position]);
                }

                Eigen::VectorXd densities = _compute_densities(distances);

                Eigen::VectorXd deltas;
                Eigen::VectorXi nn_idcs;
                std::tie(deltas, nn_idcs) = _compute_deltas(distances, densities);
                _labels = Eigen::VectorXi::Ones(deltas.rows()) * -1;

                // sort the corresponding density indices in descending order
                std::vector<size_t> idx(deltas.rows());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&deltas](const size_t lhs, const size_t rhs) {
                    return deltas(lhs) > deltas(rhs);
                });

                _centroids = Eigen::MatrixXd::Zero(K, data.cols());
                for (int i = 0; i < K; ++i) {
                    _centroids.row(i) = data.row(idx[i]);
                    _labels(idx[i]) = i;
                }

                for (uint i = 0; i < data.rows(); ++i) {
                    if (_labels(i) < 0) {
                        _labels(i) = _assign_cluster(i, nn_idcs, _labels);
                    }
                }

                _clusters.resize(K);
                for (uint i = 0; i < _labels.rows(); ++i) {
                    if (_labels(i) < 0)
                        continue;
                    _clusters[_labels(i)].conservativeResize(_clusters[_labels(i)].rows() + 1, data.cols());
                    _clusters[_labels(i)].row(_clusters[_labels(i)].rows() - 1) = data.row(i);
                }

                return _clusters;
            }

            Eigen::MatrixXi predict(const Eigen::MatrixXd& data) const
            {
            }

            void save(const tools::Archive& archive) const
            {
                archive.save(_centroids, "centroids_clusterdp");
                archive.save(_labels, "labels_clusterdp");
                for (uint i = 0; i < _clusters.size(); ++i)
                    archive.save(_clusters[i], "cluster_" + std::to_string(i) + "_data");
            }

            const Eigen::MatrixXd& centroids() const { return _centroids; }
            const Eigen::VectorXi& labels() const { return _labels; }

        protected:
            Eigen::MatrixXd _compute_distances(const Eigen::MatrixXd& data) const
            {
                // notice that we assume that the samples are stored by row
                Eigen::MatrixXd distances = Eigen::MatrixXd::Zero(data.rows(), data.rows());
                for (uint i = 0; i < data.rows(); ++i) {
                    for (uint j = i; j < data.rows(); ++j) { // only traversing half of the symmetric matrix
                        distances(i, j) = _df(data.row(i), data.row(j));
                        distances(j, i) = distances(j, i);
                    }
                }
                return distances;
            }

            Eigen::VectorXd _compute_densities(const Eigen::MatrixXd& distances) const
            {
                Eigen::VectorXd densities = Eigen::VectorXd::Zero(distances.rows());
                for (uint i = 0; i < distances.rows(); ++i) {
                    for (uint j = 0; j < distances.cols(); ++j) {
                        if (i == j) // TODO: probably right but need to check
                            continue;
                        densities(i) += _chi(distances(i, j) - _dc);
                    }
                }
                return densities;
            }

            std::pair<Eigen::VectorXd, Eigen::VectorXi> _compute_deltas(const Eigen::MatrixXd& distances, const Eigen::VectorXd densities) const
            {
                Eigen::VectorXi nn_idcs = Eigen::VectorXi::Ones(distances.rows()) * -1;
                Eigen::VectorXd deltas = Eigen::VectorXd::Zero(distances.rows());
                for (uint i = 0; i < distances.rows(); ++i) {
                    std::vector<std::pair<double, uint>> higher_density_distances;
                    for (uint j = 0; j < distances.cols(); ++j) {
                        if (densities(i) < densities(j)) {
                            higher_density_distances.push_back(std::make_pair(distances(i, j), j));
                        }
                    }

                    if (higher_density_distances.size() == 0) {
                        Eigen::VectorXd::Index max_idx;
                        deltas(i) = distances.row(i).maxCoeff(&max_idx);
                        nn_idcs(i) = max_idx;
                    }
                    else {
                        std::vector<size_t> idx(higher_density_distances.size());
                        std::iota(idx.begin(), idx.end(), 0);
                        std::sort(idx.begin(), idx.end(), [&higher_density_distances](const size_t lhs, const size_t rhs) {
                            return higher_density_distances[lhs].first < higher_density_distances[rhs].first;
                        });

                        deltas(i) = higher_density_distances[idx[0]].first;
                        nn_idcs(i) = higher_density_distances[idx[0]].second;
                    }
                }

                return std::make_pair(deltas, nn_idcs);
            }

            int _assign_cluster(uint idx, const Eigen::VectorXi& nn_idcs, const Eigen::VectorXi& labels)
            {
                if (nn_idcs(idx) < 0 && labels(idx) < 0)
                    return -1;

                if (labels(nn_idcs(idx)) >= 0) {
                    return labels(nn_idcs(idx));
                }
                else {
                    return _assign_cluster(nn_idcs(idx), nn_idcs, labels);
                }
            }

            Eigen::MatrixXd _centroids;
            std::vector<Eigen::MatrixXd> _clusters;
            Eigen::VectorXi _labels;

            DistanceFunc _df;
            DensityFunc _chi;

            double _dc;
        };

    } // namespace clustering
} // namespace aegean

#endif