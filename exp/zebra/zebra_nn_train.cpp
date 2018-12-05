#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

#include <limbo/opt/adam.hpp>

#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>

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
        BO_PARAM(int, iterations, 50000);

        /// @ingroup opt_defaults
        /// alpha - learning rate
        BO_PARAM(double, alpha, 0.001);

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
    for (int i = 0; i < num_experiments; ++i)
        position_files.push_back(path
            + "/seg_" + std::to_string(i) + "_reconstructed_positions.dat");

    std::vector<std::string> velocity_files;
    for (int i = 0; i < num_experiments; ++i)
        velocity_files.push_back(path
            + "/seg_" + std::to_string(i) + "_reconstructed_velocities.dat");

    std::vector<std::string> label_files;
    for (int i = 0; i < num_experiments; ++i)
        label_files.push_back(path
            + "/seg_" + std::to_string(i) + "_labels.dat");

    uint centroids;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_centroids.dat");
        ifs >> centroids;
        ifs.close();
    }

    uint behaviours;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_behaviours.dat");
        ifs >> behaviours;
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
        archive.load(positions, position_files[0]);
        num_individuals = positions.cols() / 2;
    }

    float timestep = static_cast<float>(centroids) / fps;
    uint aggregate_window = window_in_seconds * (fps / centroids);

    std::vector<int> sizes(behaviours, 0);
    for (uint i = 0; i < label_files.size(); ++i) {
        Eigen::MatrixXi labels;
        archive.load(labels, label_files[i]);
        for (uint b = 0; b < behaviours; ++b)
            sizes[b] += (labels.array() == static_cast<int>(b)).count() * (aggregate_window - 1) * num_individuals;
    }

    uint DIMS_OUT = 3;
    std::vector<Eigen::MatrixXd> inputs(behaviours), outputs(behaviours);
    for (uint b = 0; b < behaviours; ++b) {
        inputs[b] = Eigen::MatrixXd::Zero(sizes[b], 4 * num_individuals); // x y vx vy
        outputs[b] = Eigen::MatrixXd::Zero(sizes[b], DIMS_OUT); // dr cosdtheta sindtheta
    }

    std::vector<uint> cur_row(behaviours, 0);
    for (uint i = 0; i < position_files.size(); ++i) {
        Eigen::MatrixXd positions, velocities;
        Eigen::MatrixXi labels;
        archive.load(labels, label_files[i]);
        archive.load(positions, position_files[i]);
        archive.load(velocities, velocity_files[i]);
        Eigen::MatrixXd rolled_positions = tools::rollMatrix(positions, -1);

        uint num_individuals = positions.cols() / 2;
        assert(labels.rows() == (positions.rows() / aggregate_window) && "Dimensions don't match");

        for (uint j = 0; j < labels.rows(); ++j) {
            uint idx = j * aggregate_window;
            Eigen::MatrixXd pblock = positions.block(idx, 0, aggregate_window - 1, positions.cols());
            Eigen::MatrixXd rpblock = rolled_positions.block(idx, 0, aggregate_window - 1, rolled_positions.cols());
            Eigen::MatrixXd vblock = velocities.block(idx, 0, aggregate_window - 1, velocities.cols());

            for (uint skip = 0, ind = 0; ind < num_individuals; ++skip, ++ind) {

                Eigen::MatrixXd target(rpblock.rows(), DIMS_OUT);
                Eigen::MatrixXd pos_t(rpblock.rows(), 2);
                Eigen::MatrixXd pos_t_1(pblock.rows(), 2);

                polygons::CircularCorridor<Params> cc;
                pos_t.col(0) = rpblock.col(skip * 2).array() - cc.center().x();
                pos_t.col(1) = rpblock.col(skip * 2 + 1).array() - cc.center().y();
                pos_t_1.col(0) = pblock.col(skip * 2).array() - cc.center().x();
                pos_t_1.col(1) = pblock.col(skip * 2 + 1).array() - cc.center().y();
                Eigen::MatrixXd dradius = (pos_t.rowwise().norm() - pos_t_1.rowwise().norm());

                Eigen::MatrixXd dphi = Eigen::MatrixXd(target.rows(), 1);
                for (uint k = 0; k < dphi.rows(); ++k) {
                    double phi_t = std::fmod(std::atan2(pos_t(k, 1), pos_t(k, 0)) + 2 * M_PI, 2 * M_PI);
                    double phi_t_1 = std::fmod(std::atan2(pos_t_1(k, 1), pos_t_1(k, 0)) + 2 * M_PI, 2 * M_PI);
                    dphi(k) = (phi_t - phi_t_1);

                    // need to check the transition between 0-360 and vice versa
                    // very hacky but should work
                    if ((phi_t_1 > 3 * M_PI_2) && (phi_t < M_PI_2))
                        dphi(k) = 2 * M_PI - phi_t_1 + phi_t;

                    if ((phi_t_1 < M_PI_2) && (phi_t > 3 * M_PI_2))
                        dphi(k) = -(2 * M_PI - phi_t + phi_t_1);
                }

                target.col(0) = dradius / timestep;
                target.col(1) = dphi.array().cos() / timestep;
                target.col(2) = dphi.array().sin() / timestep;

                std::vector<uint> rm = {skip * 2, skip * 2 + 1};
                Eigen::MatrixXd reduced_pblock = pblock;
                Eigen::MatrixXd reduced_vblock = vblock;
                tools::removeCols(reduced_pblock, rm);
                tools::removeCols(reduced_vblock, rm);
                for (uint c = 0; c < pblock.rows(); ++c) {
                    Eigen::MatrixXd nrow(1, pblock.cols() + vblock.cols());
                    nrow << reduced_pblock.row(c), reduced_vblock.row(c),
                        pblock.block(c, skip * 2, 1, 2), vblock.block(c, skip * 2, 1, 2);
                    inputs[labels(j)].row(cur_row[labels(j)]) = nrow;
                    outputs[labels(j)].row(cur_row[labels(j)]) = target.row(c);
                    ++cur_row[labels(j)];
                } // c
            } // ind
        } // j
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

    if (!exists(path + std::string("/nn_info.dat"))) {
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
    }
    else {
        {
            uint num_controllers;
            std::ifstream ifs;
            ifs.open(path + "/nn_info.dat");
            ifs >> num_controllers;

            inputs.resize(num_controllers);
            outputs.resize(num_controllers);
            for (uint i = 0; i < num_controllers; ++i) {
                archive.load(inputs[i],
                    path + "/training_data_behaviour_" + std::to_string(i) + ".dat");
                archive.load(outputs[i],
                    path + "/training_labels_behaviour_" + std::to_string(i) + ".dat");
            }
        }
    }

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
            return 0.8 * simple_nn::MeanSquaredError::df(y, desired)
                + 0.2 * simple_nn::MeanSquaredError::df(y, desired_prev);
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

    int N = 1024;
    for (uint behav = 0; behav < inputs.size(); ++behav) {
        simple_nn::NeuralNet network;
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(inputs[behav].cols(), 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, outputs[behav].cols());

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
                    if ((idx % (aggregate_window - 1)) == 0)
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