#ifndef AEGEAN_ETHOGRAM_AUTOMATED_ETHOGRAM_HPP
#define AEGEAN_ETHOGRAM_AUTOMATED_ETHOGRAM_HPP

#include <clustering/kmeans.hpp>
#include <clustering/opt/gap_statistic.hpp>
#include <clustering/opt/no_opt.hpp>

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
                using clustering_opt_t = clustering::opt::NoOpt<5>;
            };

            using args = typename ae_signature::bind<A1, A2>::type;
            using ClusteringMethod =
                typename boost::parameter::binding<args, tag::clusteringmethod, typename defaults::clustering_t>::type;
            using OptMethod =
                typename boost::parameter::binding<args, tag::clusteringopt, typename defaults::clustering_opt_t>::type;

        public:
            AutomatedEthogram(const Eigen::MatrixXd& data) : _data(data)
            {
            }

            void compute()
            {
                uint optimal_k = _opt.opt_k(_data);
                _clusters = _clusterer.fit(_data, optimal_k);
            }

            const std::vector<Eigen::MatrixXd>& clusters() const { return _clusters; }
            const Eigen::VectorXi& labels() const { return _clusterer.labels(); }
            const uint num_behaviours() const { return _clusters.size(); }
            const ClusteringMethod& model() const { return _clusterer; }

        protected:
            Eigen::MatrixXd _data;

            std::vector<Eigen::MatrixXd> _clusters;
            Eigen::VectorXi _labels;

            ClusteringMethod _clusterer;
            OptMethod _opt;
        };

    } // namespace ethogram
} // namespace aegean

#endif