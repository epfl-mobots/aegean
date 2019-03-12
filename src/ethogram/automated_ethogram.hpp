#ifndef AEGEAN_ETHOGRAM_AUTOMATED_ETHOGRAM_HPP
#define AEGEAN_ETHOGRAM_AUTOMATED_ETHOGRAM_HPP

#include <clustering/kmeans.hpp>
#include <clustering/opt/gap_statistic.hpp>
#include <clustering/opt/no_opt.hpp>
#include <tools/archive.hpp>

#include <Eigen/Core>

#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/parameter.hpp>

#include <memory>

#include <iostream>

namespace aegean {
    namespace ethogram {

        BOOST_PARAMETER_TEMPLATE_KEYWORD(clusteringmethod)
        BOOST_PARAMETER_TEMPLATE_KEYWORD(clusteringopt)

        using ae_signature
            = boost::parameter::parameters<
                boost::parameter::optional<tag::clusteringmethod>,
                boost::parameter::optional<tag::clusteringopt>>;

        template <typename A1 = boost::parameter::void_, typename A2 = boost::parameter::void_>
        class AutomatedEthogram {

            struct defaults {
                using clustering_t = clustering::KMeans<aegean::defaults::KMeansPlusPlus>;
                using clustering_opt_t = clustering::opt::NoOpt<-1>;
            };

            using args = typename ae_signature::bind<A1, A2>::type;
            using ClusteringMethod =
                typename boost::parameter::binding<args, tag::clusteringmethod, typename defaults::clustering_t>::type;
            using OptMethod =
                typename boost::parameter::binding<args, tag::clusteringopt, typename defaults::clustering_opt_t>::type;

        public:
            AutomatedEthogram(const Eigen::MatrixXd& data, std::vector<uint> segment_idcs = {})
                : _data(data), _segment_idcs(segment_idcs)
            {
                if (!_segment_idcs.size()) {
                    _segment_idcs.push_back(0);
                    _segment_idcs.push_back(_data.rows());
                }
            }

            void compute(std::vector<uint> segment_idcs = {})
            {
                uint optimal_k = _opt.opt_k(_data);
                _clusters = _clusterer.fit(_data, optimal_k);
                _labels = _clusterer.labels();

                _compute_probs();
                _compute_sequence();
            }

            void save() const
            {
                _clusterer.save(_archive);

                for (uint i = 0; i < _segment_idcs.size() - 1; ++i) {
                    Eigen::MatrixXd feautre_matrix = _data.block(_segment_idcs[i], 0, _segment_idcs[i + 1] - _segment_idcs[i], _data.cols());
                    _archive.save(feautre_matrix,
                        "seg_" + std::to_string(i) + "_feature_matrix");
                }

                for (uint i = 0; i < _segment_idcs.size() - 1; ++i) {
                    Eigen::VectorXi seg_labels = _labels.block(_segment_idcs[i], 0, _segment_idcs[i + 1] - _segment_idcs[i], _labels.cols());
                    _archive.save(seg_labels,
                        "seg_" + std::to_string(i) + "_labels");
                }

                for (uint i = 0; i < _transition_probs.size(); ++i)
                    _archive.save(_transition_probs[i],
                        "seg_" + std::to_string(i) + "_transition_probabilities");

                for (uint i = 0; i < _behaviour_probs.size(); ++i)
                    _archive.save(_behaviour_probs[i],
                        "seg_" + std::to_string(i) + "_behaviour_probabilities");

                for (uint i = 0; i < _behaviour_sequence.size(); ++i)
                    _archive.save(_behaviour_sequence[i],
                        "seg_" + std::to_string(i) + "_behaviour_sequence");
            }

            const std::vector<Eigen::MatrixXd>& clusters() const { return _clusters; }
            const Eigen::VectorXi& labels() const { return _labels; }
            const uint num_behaviours() const { return _clusters.size(); }
            const ClusteringMethod& model() const { return _clusterer; }
            const std::string& etho_path() const { return _archive.dir_name(); }
            const tools::Archive& archive() const { return _archive; }

            const std::vector<Eigen::MatrixXd>& transition_probs() const { return _transition_probs; }
            const std::vector<Eigen::VectorXd>& behaviour_probs() const { return _behaviour_probs; }

        protected:
            void _compute_probs()
            {
                _transition_probs.clear();
                _behaviour_probs.clear();

                for (uint el = 0; el < _segment_idcs.size() - 1; ++el) {
                    Eigen::VectorXi seg_labels = _labels.block(_segment_idcs[el], 0, _segment_idcs[el + 1] - _segment_idcs[el], _labels.cols());

                    Eigen::MatrixXd occurence_graph = Eigen::MatrixXd::Zero(num_behaviours(), num_behaviours());
                    for (uint i = 1; i < seg_labels.rows(); ++i)
                        occurence_graph(seg_labels(i - 1), seg_labels(i)) += 1;

                    Eigen::VectorXd bprobs(num_behaviours());
                    for (int i = 0; i < static_cast<int>(num_behaviours()); ++i)
                        bprobs(i) = (seg_labels.array() == i).count() / static_cast<double>(seg_labels.rows());
                    _behaviour_probs.push_back(bprobs);

                    Eigen::MatrixXd tprobs(num_behaviours(), num_behaviours());
                    for (uint i = 0; i < occurence_graph.rows(); ++i)
                        tprobs.row(i) = occurence_graph.row(i) / occurence_graph.row(i).sum();
                    _transition_probs.push_back(tprobs);
                }
            }

            void _compute_sequence()
            {
                _behaviour_sequence.clear();

                for (uint el = 0; el < _segment_idcs.size() - 1; ++el) {
                    Eigen::MatrixXi seq_mat = Eigen::MatrixXi::Zero(1, 3); // cols -> cluster timestep_start timestep_end
                    Eigen::VectorXi seg_labels = _labels.block(_segment_idcs[el], 0, _segment_idcs[el + 1] - _segment_idcs[el], _labels.cols());

                    seq_mat(0, 0) = seg_labels(0);
                    seq_mat(0, 1) = 0;
                    for (uint i = 1; i < seg_labels.rows(); ++i) {
                        if (seg_labels(i - 1) != seg_labels(i)) {
                            seq_mat(seq_mat.rows() - 1, 2) = i - 1;
                            seq_mat.conservativeResize(seq_mat.rows() + 1, seq_mat.cols());
                            seq_mat(seq_mat.rows() - 1, 0) = seg_labels(i);
                            seq_mat(seq_mat.rows() - 1, 1) = i;
                        }
                    }

                    if (seq_mat(seq_mat.rows() - 1, 1) == seg_labels(seq_mat.rows() - 1)) {
                        seq_mat(seq_mat.rows() - 1, 2) = seg_labels.rows() - 1;
                    }
                    else {
                        seq_mat(seq_mat.rows() - 1, 2) = seg_labels.rows() - 2;
                        seq_mat.conservativeResize(seq_mat.rows() + 1, seq_mat.cols());
                        seq_mat(seq_mat.rows() - 1, 0) = seg_labels(seq_mat.rows() - 1);
                        seq_mat(seq_mat.rows() - 1, 1) = seg_labels.rows() - 1;
                        seq_mat(seq_mat.rows() - 1, 2) = seg_labels.rows() - 1;
                    }
                    _behaviour_sequence.push_back(seq_mat);
                }
            }

            Eigen::MatrixXd _data;
            std::vector<uint> _segment_idcs;

            std::vector<Eigen::MatrixXd> _clusters;
            Eigen::VectorXi _labels;

            ClusteringMethod _clusterer;
            OptMethod _opt;

            tools::Archive _archive;

            std::vector<Eigen::MatrixXd> _transition_probs;
            std::vector<Eigen::VectorXd> _behaviour_probs;
            std::vector<Eigen::MatrixXi> _behaviour_sequence;
        };

    } // namespace ethogram
} // namespace aegean

#endif