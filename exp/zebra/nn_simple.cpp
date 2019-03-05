#include "nn_structure.hpp"

#include <limbo/opt/adam.hpp>

#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>

#include <features/inter_individual_distance.hpp>
#include <features/linear_velocity.hpp>
#include <features/angular_velocity.hpp>
#include <features/distance_to_agents.hpp>
#include <features/angle_difference.hpp>
#include <features/linear_velocity_difference.hpp>
#include <features/angular_velocity_difference.hpp>
#include <features/radial_velocity.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace aegean;
using namespace tools;

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };

    struct opt_adam {
        /// @ingroup opt_defaults
        /// number of max iterations
        BO_PARAM(int, iterations, 100000);

        /// @ingroup opt_defaults
        /// alpha - learning rate
        BO_PARAM(double, alpha, 0.0005);

        /// @ingroup opt_defaults
        /// β1
        BO_PARAM(double, b1, 0.9);

        /// @ingroup opt_defaults
        /// β2
        BO_PARAM(double, b2, 0.999);

        /// @ingroup opt_defaults
        /// norm epsilon for stopping
        BO_PARAM(double, eps_stop, 0.0);
    };
};

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> construct_nn_sets(int argc, char** argv)
{
    Archive archive(false);
    std::string path(argv[1]);
    int num_experiments = std::stoi(argv[2]);

    std::vector<std::string> position_files;
    std::vector<std::string> velocity_files;
    std::vector<std::string> acceleration_files;
    for (int i = 0; i < num_experiments; ++i) {
        position_files.push_back(path + "/seg_" + std::to_string(i) + "_reconstructed_positions.dat");
        velocity_files.push_back(path + "/seg_" + std::to_string(i) + "_reconstructed_velocities.dat");
        acceleration_files.push_back(path + "/seg_" + std::to_string(i) + "_reconstructed_accelerations.dat");
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

    uint num_individuals, duration;
    {
        Eigen::MatrixXd positions;
        archive.load(positions, position_files[0]);
        num_individuals = positions.cols() / 2;
        duration = positions.rows();
    }

    // Below we initialize all the input features that will be necessary for the net
    float timestep = static_cast<float>(centroids) / fps;

    // additional features to use as inputs
    // using circular_corridor_t = polygons::CircularCorridor<Params>;
    // using distance_func_t = defaults::distance_functions::angular<circular_corridor_t>;
    // // using euc_distance_func_t = defaults::distance_functions::euclidean;
    // circular_corridor_t cc;

    // set the dims we just calculated
    std::vector<Eigen::MatrixXd>
        inputs(1, Eigen::MatrixXd::Zero(position_files.size() * duration * num_individuals, nn_strucutre::DIMS_IN)),
        outputs(1, Eigen::MatrixXd::Zero(position_files.size() * duration * num_individuals, nn_strucutre::DIMS_OUT));

    // restructure the data in the correct input form
    // of the nns for each experimental file
    int cur_row = 0;
    for (uint i = 0; i < position_files.size(); ++i) {
        Eigen::MatrixXd positions, velocities, accelerations;
        archive.load(positions, position_files[i]);
        archive.load(velocities, velocity_files[i]);
        archive.load(accelerations, acceleration_files[i]);

        Eigen::MatrixXd rolled_positions = tools::rollMatrix(positions, -1);
        Eigen::MatrixXd rolled_velocities = tools::rollMatrix(velocities, -1);
        Eigen::MatrixXd rolled_accelerations = tools::rollMatrix(accelerations, -1);

        for (uint ind = 0; ind < num_individuals; ++ind) {
            Eigen::MatrixXd pos_t = rolled_positions.block(0, ind * 2, duration, 2);
            Eigen::MatrixXd pos_t_1 = positions.block(0, ind * 2, duration, 2);
            Eigen::MatrixXd vel_t = rolled_velocities.block(0, ind * 2, duration, 2);
            Eigen::MatrixXd vel_t_1 = velocities.block(0, ind * 2, duration, 2);
            Eigen::MatrixXd acc_t = rolled_accelerations.block(0, ind * 2, duration, 2);
            Eigen::MatrixXd acc_t_1 = accelerations.block(0, ind * 2, duration, 2);

            Eigen::MatrixXd dpos = pos_t - pos_t_1;
            Eigen::MatrixXd dvel = vel_t - vel_t_1;

            Eigen::MatrixXd target = vel_t;

            // remove columns corresponding to the excluded individual
            std::vector<uint> rm = {ind * 2, ind * 2 + 1};
            Eigen::MatrixXd reduced_pos_block = positions;
            Eigen::MatrixXd reduced_vel_block = velocities;
            Eigen::MatrixXd reduced_acc_block = accelerations;

            tools::removeCols(reduced_pos_block, rm);
            tools::removeCols(reduced_vel_block, rm);
            tools::removeCols(reduced_acc_block, rm);

            // split the resulting data into inputs and outputs
            for (uint ts = 0; ts < duration; ++ts) {
                Eigen::MatrixXd nrow(1, inputs[0].cols());
                nrow <<
                    // net inputs
                    reduced_pos_block.block(ts, 0, 1, reduced_pos_block.cols()),
                    reduced_vel_block.block(ts, 0, 1, reduced_vel_block.cols()),
                    pos_t_1.block(ts, 0, 1, 2),
                    vel_t_1.block(ts, 0, 1, 2)
                    // net inputs
                    ;
                inputs[0].row(cur_row) = nrow;
                outputs[0].row(cur_row) = target.row(ts);
                ++cur_row;
            }
        } // ind
    } // i

    return std::make_pair(inputs, outputs);
}

bool exists(std::string filename)
{
    std::ifstream ifs(filename);
    return ifs.good();
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> load(int argc, char** argv)
{
    std::string path(argv[1]);
    Archive archive(false);
    std::vector<Eigen::MatrixXd> inputs, outputs;

    // if (!exists(path + std::string("/nn_info.dat"))) {
    std::tie(inputs, outputs) = construct_nn_sets(argc, argv);
    for (uint i = 0; i < inputs.size(); ++i) {
        archive.save(inputs[i],
            path + "/training_data_behaviour_" + std::to_string(i));
        archive.save(outputs[i],
            path + "/training_labels_behaviour_" + std::to_string(i));
    }

    // write nn dims in a file to aid the loading
    // TODO: store nn structure instead
    {
        std::ofstream ofs(path + "/nn_info.dat");
        ofs << outputs.size() << std::endl;
    }
    // }
    // else {
    //     {
    //         uint num_controllers;
    //         std::ifstream ifs;
    //         ifs.open(path + "/nn_info.dat");
    //         ifs >> num_controllers;

    //         inputs.resize(num_controllers);
    //         outputs.resize(num_controllers);
    //         for (uint i = 0; i < num_controllers; ++i) {
    //             archive.load(inputs[i],
    //                 path + "/training_data_behaviour_" + std::to_string(i) + ".dat");
    //             archive.load(outputs[i],
    //                 path + "/training_labels_behaviour_" + std::to_string(i) + ".dat");
    //         }
    //     }
    // }

    return std::make_pair(inputs, outputs);
}

namespace aegean {
    struct SmoothTrajectory : public simple_nn::MeanSquaredError {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            uint split = y_d.rows() / 2;
            Eigen::MatrixXd desired = y_d.block(0, 0, split, y_d.cols());
            Eigen::MatrixXd desired_prev = y_d.block(split, 0, split, y_d.cols());
            return simple_nn::MeanSquaredError::f(y, desired);
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            uint split = y_d.rows() / 2;
            Eigen::MatrixXd desired = y_d.block(0, 0, split, y_d.cols());
            Eigen::MatrixXd desired_prev = y_d.block(split, 0, split, y_d.cols());
            return 0.5 * simple_nn::MeanSquaredError::df(y, desired)
                + 0.5 * simple_nn::MeanSquaredError::df(y, desired_prev);
        }
    };
} // namespace aegean

#ifndef SMOOTH_LOSS
using loss_t = simple_nn::MeanSquaredError;
#else
using loss_t = aegean::SmoothTrajectory;
#endif

int main(int argc, char** argv)
{
    std::srand(std::time(NULL));

    // TODO: add option parser
    assert(argc == 3);
    std::vector<Eigen::MatrixXd> inputs, outputs;
    std::tie(inputs, outputs) = load(argc, argv);
    std::string path(argv[1]);
    Archive archive(false);

    uint num_individuals;
    {
        Eigen::MatrixXd positions;
        archive.load(positions, path + "/seg_0_reconstructed_positions.dat");
        num_individuals = static_cast<uint>(positions.cols() / 2);
    }

#ifdef SMOOTH_LOSS
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
#endif

    int N = 512;
    for (uint behav = 0; behav < inputs.size(); ++behav) {
        simple_nn::NeuralNet network;
        nn_strucutre::init_nn(network, num_individuals);

        // Random initial weights
        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());
        network.set_weights(theta);

        int epoch_count = 0;
        auto func = [&](const Eigen::VectorXd& params, bool eval_grad) {
            assert(eval_grad);

            Eigen::MatrixXd samples(inputs[behav].cols(), N);
            limbo::tools::rgen_int_t rgen(0, inputs[behav].rows() - 1);

#ifndef SMOOTH_LOSS
            Eigen::MatrixXd observations(outputs[behav].cols(), N);

            for (int i = 0; i < N; i++) {
                int idx = rgen.rand();
                samples.col(i) = inputs[behav].transpose().col(idx);
                observations.col(i) = outputs[behav].transpose().col(idx);
            }
#else
            Eigen::MatrixXd observations(outputs[behav].cols() * 2, N);

            for (int i = 0; i < N; i++) {
                int idx = -1;
                while (idx < 0) {
                    idx = rgen.rand();
                    if ((idx % (aggregate_window)) == 0)
                        idx = -1;
                }
                samples.col(i) = inputs[behav].transpose().col(idx);
                observations.col(i) << outputs[behav].transpose().col(idx),
                    outputs[behav].transpose().col(idx - 1);
            }
#endif

            network.set_weights(params);

            double f = -network.get_loss<loss_t>(samples, observations);

            // f += -params.norm();
            Eigen::VectorXd grad = -network.backward<loss_t>(samples, observations);
            // grad.array() -= 2 * params.array();

            if (epoch_count++ % 1000 == 0)
                std::cout << "Loss (iteration " << epoch_count - 1 << "): " << -f << std::endl;

            return limbo::opt::eval_t{f, grad};
        };
        limbo::opt::Adam<Params> adam;
        Eigen::VectorXd best_theta = adam(func, theta, false);

#ifndef SMOOTH_LOSS
        double f = network.get_loss<loss_t>(inputs[behav].transpose(), outputs[behav].transpose());
        std::cout << "Loss: " << f << std::endl;
#else
        int skip = outputs[behav].cols() / (aggregate_window - 1);
        Eigen::MatrixXd e_outputs(outputs[behav].rows() - skip, outputs[behav].cols() * 2);
        for (uint i = 0; i < outputs[behav].rows(); ++i) {
            if ((i % (aggregate_window - 1)) == 0)
                continue;
            e_outputs.row(i) << outputs[behav].row(i), outputs[behav].row(i - 1);
        }

        double f = network.get_loss<loss_t>(inputs[behav].transpose(), e_outputs.transpose());
        std::cout << "Loss: " << f << std::endl;
#endif

        archive.save(network.weights(),
            path + "/nn_controller_weights_" + std::to_string(behav));
    }

    return 0;
}