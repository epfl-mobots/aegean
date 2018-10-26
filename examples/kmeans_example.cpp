#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>

#include <tools/reconstruction/cspace.hpp>
#include <tools/polygons/circular_corridor.hpp>

#include <iostream>
#include <vector>
#include <Eigen/Core>

using namespace aegean;
using namespace tools;

struct Params {
    struct CircularCorridor : public polygons::defaults::CircularCorridor {
    };
};

int main()
{
    std::vector<uint> prob_cols = {2, 5, 8, 11, 14, 17};
    Eigen::MatrixXd data;
    Archive dl;
    dl.load<Eigen::MatrixXd>(
        data,
        "/home/vpapaspy/Workspaces/mobots_ws/Research/Ethograms/aegean/experiments/zebrafish/"
        "data/fish_only/4_fish_only/trajectories.txt",
        1);
    removeCols(data, prob_cols);
    uint start_idx = data.rows() - 27855;
    Eigen::MatrixXd positions = data.block(start_idx, 0, 27855, data.cols());

    ExperimentDataFrame<reconstruction::CSpace<polygons::CircularCorridor<Params>>> itd(
        positions, 15, 3, 1.13 / 1024, 855);
    // std::cout << itd.positions().rows() << std::endl;

    return 0;
}