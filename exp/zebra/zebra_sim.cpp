#include "sim/aegean_simulation.hpp"
#include "sim/aegean_individual.hpp"

#include <simple_nn/neural_net.hpp>
#include <clustering/kmeans.hpp>

#include <tools/archive.hpp>
#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>
#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

using namespace aegean;
using namespace tools;

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };
};

int main(int argc, char** argv)
{
    assert(argc == 4);
    std::string path(argv[1]);
    int exp_num = std::stoi(argv[2]);
    int exclude_idx = std::stoi(argv[3]);
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

    // creating the nn structure
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
    std::shared_ptr<Eigen::MatrixXd> predictions = std::make_shared<Eigen::MatrixXd>();

    simu::simulation::AegeanSimulation sim(nn,
        std::make_shared<aegean::clustering::KMeans<>>(km),
        std::make_shared<Eigen::MatrixXd>(positions),
        std::make_shared<Eigen::MatrixXd>(velocities),
        predictions,
        generated_positions,
        {exclude_idx});
    sim.aegean_sim_settings().aggregate_window = aggregate_window;
    sim.aegean_sim_settings().timestep = timestep;
    sim.sim_time() = positions.rows();
    // sim.spin_once();
    sim.spin();

    archive.save(predictions->col(exclude_idx),
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx));

    Eigen::MatrixXd extended_traj(generated_positions->rows(), generated_positions->cols());
    archive.save(extended_traj,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_positions");

    using distance_func_t
        = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    features::InterIndividualDistance<distance_func_t> iid;
    iid(extended_traj, timestep);

    features::Alignment align;
    align(extended_traj, timestep);

    Eigen::MatrixXd feature_matrix(iid.get().rows(), iid.get().cols() + align.get().cols());
    feature_matrix << iid.get(), align.get();

    Eigen::MatrixXd avg_fm;
    for (uint i = 0; i < feature_matrix.rows(); i += aggregate_window) {
        avg_fm.conservativeResize(avg_fm.rows() + 1, feature_matrix.cols());
        avg_fm.row(avg_fm.rows() - 1) = feature_matrix.block(i, 0, aggregate_window, feature_matrix.cols()).colwise().mean();
    }

    archive.save(feature_matrix.col(0),
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_interindividual");

    archive.save(feature_matrix.col(1),
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_alignment");

    archive.save(avg_fm,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_feature_matrix");

    return 0;
}