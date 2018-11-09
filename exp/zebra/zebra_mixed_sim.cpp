#include <simple_nn/neural_net.hpp>

#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <tools/mathtools.hpp>

#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>

#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace aegean;
using namespace tools;

#define USE_POLAR

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };
};

int main(int argc, char** argv)
{
    std::srand(std::time(NULL));

    // TODO: add option parser
    assert(argc == 4);
    std::string path(argv[1]);
    int exp_num = std::stoi(argv[2]);
    int exclude_idx = std::stoi(argv[3]);
    Archive archive(false);

    uint behaviours;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_behaviours.dat");
        ifs >> behaviours;
        ifs.close();
    }

    uint centroids;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_centroids.dat");
        ifs >> centroids;
        ifs.close();
    }

    uint fps;
    {
        std::ifstream ifs;
        ifs.open(path + "/fps.dat");
        ifs >> fps;
        ifs.close();
    }

    uint window_in_seconds;
    {
        std::ifstream ifs;
        ifs.open(path + "/window_in_seconds.dat");
        ifs >> window_in_seconds;
        ifs.close();
    }

    uint aggregate_window = window_in_seconds * (fps / centroids);

    Eigen::MatrixXd positions, velocities, labels;
    archive.load(positions, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_positions.dat");
    archive.load(velocities, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_velocities.dat");
    archive.load(labels, path + "/seg_" + std::to_string(exp_num) + "_labels.dat");

    Eigen::MatrixXd removed(positions.rows(), 2);
    removed.col(0) = positions.col(exclude_idx * 2);
    removed.col(1) = positions.col(exclude_idx * 2 + 1);

    std::vector<simple_nn::NeuralNet> nn(behaviours);
    for (uint b = 0; b < behaviours; ++b) {
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(
            positions.cols() - 2 + velocities.cols() - 2, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 3); // pos_in_cc phi.cos phi.sin
        Eigen::MatrixXd weights;
        archive.load(weights, path + "/nn_controller_weights_" + std::to_string(b) + ".dat");
        nn[b].set_weights(weights);
    }

    std::vector<uint> rm = {static_cast<uint>(exclude_idx * 2), static_cast<uint>(exclude_idx * 2 + 1)};
    tools::removeCols(positions, rm);
    tools::removeCols(velocities, rm);

    Eigen::MatrixXd predictions(positions.rows(), 2);
    for (uint i = 0; i < positions.rows(); ++i) {
        Eigen::MatrixXd sample(1, positions.cols() + velocities.cols());
        sample << positions.row(i), velocities.row(i);
        uint label_idx = static_cast<int>(i / aggregate_window);
        Eigen::MatrixXd pred = nn[labels(label_idx)].forward(sample.transpose()).transpose();

        polygons::CircularCorridor<Params> cc;
        double radius = (pred(0) + 1) / (2 * 10) + cc.inner_radius();
        double phi = std::atan2(pred(2), pred(1));
        predictions(i, 0) = radius * std::cos(phi) + cc.center().x();
        predictions(i, 1) = radius * std::sin(phi) + cc.center().y();
    }

    Eigen::MatrixXd rolled = tools::rollMatrix(predictions, -1);
    rolled.row(rolled.rows() - 1) = removed.row(rolled.rows() - 1);
    double error = (removed - predictions).rowwise().norm().mean();

    {
        std::cout << "Trajectory mean distance: " << error << std::endl;
        std::ofstream ofs(
            path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_error_norm.dat");
        ofs << error << std::endl;
    }
    {
        std::cout << "Mean body length distance: " << error / 4. << std::endl;
        std::ofstream ofs(
            path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_error_body_length.dat");
        ofs << error << std::endl;
    }

    archive.save(predictions,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx));

    Eigen::MatrixXd extended_traj(positions.rows(), positions.cols() + rolled.cols());
    extended_traj << positions, rolled;
    archive.save(predictions,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_positions");

    using distance_func_t
        = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    features::InterIndividualDistance<distance_func_t> iid;
    iid(positions, static_cast<float>(centroids) / fps);

    features::Alignment align;
    align(positions, static_cast<float>(centroids) / fps);

    Eigen::MatrixXd feature_matrix(iid.get().rows(), iid.get().cols() + align.get().cols());
    feature_matrix << iid.get(), align.get();

    Eigen::MatrixXd avg_fm;
    for (uint i = 0; i < feature_matrix.rows(); i += aggregate_window) {
        avg_fm.conservativeResize(avg_fm.rows() + 1, feature_matrix.cols());
        avg_fm.row(avg_fm.rows() - 1) = feature_matrix.block(i, 0, aggregate_window, feature_matrix.cols()).colwise().mean();
    }

    archive.save(avg_fm.col(0),
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_interindividual");

    archive.save(avg_fm.col(1),
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_alignment");

    archive.save(avg_fm,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_feature_matrix");

    return 0;
}