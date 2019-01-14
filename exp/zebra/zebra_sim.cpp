#include "sim/aegean_simulation.hpp"

#include <simple_nn/neural_net.hpp>
#include <tools/archive.hpp>

using namespace aegean;
using namespace tools;

int main(int argc, char** argv)
{

    std::string path(argv[1]);
    int exp_num = std::stoi(argv[2]);
    // int exclude_idx = std::stoi(argv[3]);
    Archive archive(false);

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
    float timestep = static_cast<float>(centroids) / fps;

    Eigen::MatrixXd positions, velocities;
    // std::shared_ptr<Eigen::MatrixXd> positions = std::make_shared<Eigen::MatrixXd>();
    // std::shared_ptr<Eigen::MatrixXd> velocities = std::make_shared<Eigen::MatrixXd>();
    archive.load(positions, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_positions.dat");
    archive.load(velocities, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_velocities.dat");

    simple_nn::NeuralNet network;
    // TODO: add layers here

    simu::simulation::AegeanSimulation sim(network, std::make_shared<Eigen::MatrixXd>(positions), std::make_shared<Eigen::MatrixXd>(velocities));
    // sim.spin_once();
    sim.spin();

    return 0;
}