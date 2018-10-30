
#include <clustering/kmeans.hpp>
#include <clustering/opt/gap_statistic.hpp>

#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>
#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>

#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace aegean;
using namespace clustering;
using namespace tools;
using namespace opt;

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };
};

int main()
{
    Eigen::MatrixXd data;
    Archive dl;
    dl.load<Eigen::MatrixXd>(
        data,
        "/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/experiments/zebrafish/"
        "data/fish_only/3_fish_only/trajectories.txt",
        1);
    std::vector<uint> prob_cols = {2, 5, 8, 11, 14, 17};
    removeCols(data, prob_cols); // remove idTracker probability cols
    uint start_idx
        = data.rows() - 27855; // bring all trajectories to the same length for comparison
    Eigen::MatrixXd positions = data.block(start_idx, 0, 27855, data.cols());

    // using distance_func_t = aegean::defaults::distance_functions::euclidean;
    using distance_func_t
        = aegean::defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    using reconstruction_t = reconstruction::CSpace<polygons::CircularCorridor<Params>>;
    using features_t
        = boost::fusion::vector<aegean::features::InterIndividualDistance<distance_func_t>,
            aegean::features::Alignment>;
    ExperimentDataFrame<reconfun<reconstruction_t>, featset<features_t>> edf(positions, 15, 3,
        1.13 / 1024, 855);
    Eigen::MatrixXd fm = edf.get_feature_matrix();

    GapStatistic<KMeans<defaults::KMeansPlusPlus>> gs;
    uint optimal_k = gs.opt_k(fm, 2, 10);

    KMeans<defaults::KMeansPlusPlus> km;
    std::vector<Eigen::MatrixXd> clusters = km.fit(optimal_k, 3);
    std::cout << km.centroids() << std::endl;

    return 0;
}