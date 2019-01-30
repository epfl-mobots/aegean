#include "sim/aegean_simulation.hpp"
#include "sim/aegean_individual.hpp"

#include "nn_structure.hpp"

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
    // load experiment parameters
#ifndef WITH_CMAES_SIM
    assert(argc == 4);
#else
    assert(argc == 5);
#endif
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

    uint num_individuals;
    {
        Eigen::MatrixXd positions;
        archive.load(positions, path + "/seg_0_reconstructed_positions.dat");
        num_individuals = static_cast<uint>(positions.cols() / 2);
    }

    uint aggregate_window = window_in_seconds * (fps / num_centroids);
    std::cout << "done" << std::endl;
    float timestep = static_cast<float>(num_centroids) / fps;

    // creating the nn structure
    simu::simulation::NNVec nn(num_behaviours);
    nn_strucutre::init_nn(nn, num_behaviours, num_individuals);

#ifndef WITH_CMAES_SIM
    for (uint b = 0; b < num_behaviours; ++b) {
        Eigen::MatrixXd weights;
        archive.load(weights, path + "/nn_controller_weights_" + std::to_string(b) + ".dat");
        nn[b]->set_weights(weights);
    }
#else
    Eigen::MatrixXd params;
    archive.load(params, path + "/cmaes_iter_" + std::string(argv[4]) + "_weights.dat");
    for (uint b = 0; b < num_behaviours; ++b) {
        Eigen::MatrixXd behaviour_specific_params;
        behaviour_specific_params = params.block(
            static_cast<int>(params.rows() / num_behaviours) * b, 0,
            static_cast<int>(params.rows() / num_behaviours), params.cols());
        nn[b]->set_weights(behaviour_specific_params);
    }
#endif

    // we load the experiment positions, velocities, etc
    // and wrap them in shared pointers to avoid continuous
    // context switching in the simulation (we have very big matrices)
    Eigen::MatrixXd positions, velocities, cluster_centers;
    archive.load(positions, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_positions.dat");
    archive.load(velocities, path + "/seg_" + std::to_string(exp_num) + "_reconstructed_velocities.dat");
    archive.load(cluster_centers, path + "/centroids_kmeans.dat");
    aegean::clustering::KMeans<> km(cluster_centers);

    std::shared_ptr<Eigen::MatrixXd> generated_positions = std::make_shared<Eigen::MatrixXd>();
    std::shared_ptr<Eigen::MatrixXd> generated_velocities = std::make_shared<Eigen::MatrixXd>();
    std::shared_ptr<Eigen::MatrixXd> predictions = std::make_shared<Eigen::MatrixXd>();

    simu::simulation::AegeanSimulation sim(nn,
        std::make_shared<aegean::clustering::KMeans<>>(km),
        std::make_shared<Eigen::MatrixXd>(positions),
        std::make_shared<Eigen::MatrixXd>(velocities),
        predictions,
        generated_positions,
        generated_velocities,
        {exclude_idx});
    sim.aegean_sim_settings().aggregate_window = aggregate_window;
    sim.aegean_sim_settings().timestep = timestep;
    sim.sim_time() = positions.rows();
    // sim.spin_once();
    sim.spin();

    // store the output results
    archive.save(predictions->col(exclude_idx),
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx));

    archive.save(*generated_positions,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx) + "_extended_positions");

    using distance_func_t
        = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    features::InterIndividualDistance<distance_func_t> iid;
    iid(*generated_positions, timestep);

    features::Alignment align;
    align(*generated_positions, timestep);

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