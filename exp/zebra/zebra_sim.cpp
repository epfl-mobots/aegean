#include "sim/aegean_simulation.hpp"

#include <simple_nn/neural_net.hpp>
#include <tools/archive.hpp>
#include <clustering/kmeans.hpp>

using namespace aegean;
using namespace tools;

int main(int argc, char** argv)
{

    std::string path(argv[1]);
    int exp_num = std::stoi(argv[2]);
    // int exclude_idx = std::stoi(argv[3]);
    Archive archive(false);

    uint num_behaviours;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_behaviours.dat");
        ifs >> num_behaviours;
        ifs.close();
    }

    uint num_centroids;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_centroids.dat");
        ifs >> num_centroids;
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

    uint aggregate_window = window_in_seconds * (fps / num_centroids);
    float timestep = static_cast<float>(num_centroids) / fps;

    simu::simulation::NNVec nn(num_behaviours);
    for (uint b = 0; b < num_behaviours; ++b) {
        nn[b] = std::make_shared<simple_nn::NeuralNet>();
        nn[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(10 + 10 + 1 + 2 + 2, 100);
        nn[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 100);
        nn[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 100);
        nn[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Linear>>(100, 2);
        Eigen::MatrixXd weights;
        archive.load(weights, path + "/nn_controller_weights_" + std::to_string(b) + ".dat");
        nn[b]->set_weights(weights);
    }

    Eigen::MatrixXd positions, velocities, cluster_centers;
    archive.load(positions, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_positions.dat");
    archive.load(velocities, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_velocities.dat");
    archive.load(cluster_centers, path + "/centroids_kmeans.dat");
    aegean::clustering::KMeans<> km(cluster_centers);

    std::shared_ptr<Eigen::MatrixXd> generated_positions = std::make_shared<Eigen::MatrixXd>();

    simu::simulation::AegeanSimulation sim(nn,
        std::make_shared<aegean::clustering::KMeans<>>(km),
        std::make_shared<Eigen::MatrixXd>(positions),
        std::make_shared<Eigen::MatrixXd>(velocities),
        generated_positions,
        {0});
    sim.aegean_sim_settings().aggregate_window = aggregate_window;
    sim.aegean_sim_settings().timestep = timestep;
    // sim.spin_once();
    sim.spin();

    std::cout << generated_positions->topRows(10) << std::endl;

    return 0;
}