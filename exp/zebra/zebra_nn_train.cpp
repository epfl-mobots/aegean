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

#define USE_POLAR

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };
};

template <typename CircularCorridor = polygons::CircularCorridor<Params>>
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

    uint aggregate_window = window_in_seconds * (fps / centroids);

    std::vector<int> sizes(behaviours, 0);
    for (uint i = 0; i < label_files.size(); ++i) {
        Eigen::MatrixXi labels;
        archive.load(labels, label_files[i]);
        for (uint b = 0; b < behaviours; ++b)
            sizes[b] += (labels.array() == static_cast<int>(b)).count() * (aggregate_window - 1) * num_individuals;
    }

    std::vector<Eigen::MatrixXd> inputs(behaviours), outputs(behaviours);
    for (uint b = 0; b < behaviours; ++b) {
        inputs[b] = Eigen::MatrixXd::Zero(sizes[b], 4 * (num_individuals - 1)); // x y vx vy
        outputs[b] = Eigen::MatrixXd::Zero(sizes[b], 2); // x y
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
                Eigen::MatrixXd target(rpblock.rows(), 2);
                target.col(0) = rpblock.col(skip * 2);
                target.col(1) = rpblock.col(skip * 2 + 1);

#ifdef USE_POLAR
                CircularCorridor cc;
                target.col(0) = target.col(0).unaryExpr([&](double val) {
                    return val - cc.center().x();
                });
                target.col(1) = target.col(1).unaryExpr([&](double val) {
                    return val - cc.center().y();
                });
                Eigen::MatrixXd radius = target.rowwise().norm();
                Eigen::MatrixXd phi = Eigen::MatrixXd(target.rows(), 1);
                for (uint k = 0; k < phi.rows(); ++k)
                    phi(k) = std::fmod(std::atan2(target(k, 1), target(k, 0)) * 180 / M_PI + 360, 360) / 360;
                target.col(0) = radius;
                target.col(0) = phi;
#endif

                std::vector<uint> rm = {skip * 2, skip * 2 + 1};
                Eigen::MatrixXd reduced_pblock = pblock;
                tools::removeCols(reduced_pblock, rm);
                Eigen::MatrixXd reduced_vblock = vblock;
                tools::removeCols(reduced_vblock, rm);

                for (uint c = 0; c < reduced_pblock.rows(); ++c) {
                    Eigen::MatrixXd nrow(1, reduced_pblock.cols() + reduced_vblock.cols());
                    nrow << reduced_pblock.row(c), reduced_vblock.row(c);
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

int main(int argc, char** argv)
{
    // TODO: add option parser
    assert(argc == 3);
    std::vector<Eigen::MatrixXd> inputs, outputs;
    std::tie(inputs, outputs) = load(argc, argv);

    return 0;
}