#include <simple_nn/neural_net.hpp>

#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <tools/mathtools.hpp>

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
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Sigmoid>>(
            positions.cols() - 2 + velocities.cols() - 2, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Sigmoid>>(100, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Sigmoid>>(100, 2); // pos_in_cc phi
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
        predictions.row(i) = nn[labels(label_idx)].forward(sample.transpose()).transpose();
#ifdef USE_POLAR
        polygons::CircularCorridor<Params> cc;
        double radius = predictions(i, 0) / 10 + cc.inner_radius();
        double phi = predictions(i, 1) * 360.;
        if (phi > 180)
            phi -= 360;
        phi *= M_PI / 180.;
        predictions(i, 0) = radius * std::cos(phi) + cc.center().x();
        predictions(i, 1) = radius * std::sin(phi) + cc.center().y();
#endif
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

    return 0;
}