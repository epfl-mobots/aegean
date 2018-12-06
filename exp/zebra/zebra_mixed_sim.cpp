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
#include <cmath>

using namespace aegean;
using namespace tools;

#ifndef USE_ORIGINAL_LABELS
#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>
#include <clustering/kmeans.hpp>

using namespace clustering;

#endif

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
    float timestep = static_cast<float>(centroids) / fps;

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
            positions.cols() + velocities.cols() - 2 + 1 + 2, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Linear>>(100, 2); // dr cosdphi sindphi
        Eigen::MatrixXd weights;
        archive.load(weights, path + "/nn_controller_weights_" + std::to_string(b) + ".dat");
        nn[b].set_weights(weights);
    }

#ifndef USE_ORIGINAL_LABELS
    Eigen::MatrixXd c;
    archive.load(c, path + "/centroids_kmeans.dat");
    KMeans<> km(c);
#endif

    Eigen::MatrixXd e_positions(positions.rows() + 1, positions.cols());
    Eigen::MatrixXd e_velocities(velocities.rows() + 1, velocities.cols());
    e_positions.block(0, 0, positions.rows(), positions.cols()) = positions;
    e_velocities.block(0, 0, velocities.rows(), velocities.cols()) = velocities;
    e_positions.row(0) = positions.row(0);
    e_velocities.row(0) = e_velocities.row(0);

    std::vector<uint> rm = {static_cast<uint>(exclude_idx * 2), static_cast<uint>(exclude_idx * 2 + 1)};
    tools::removeCols(positions, rm);
    tools::removeCols(velocities, rm);

    double x = 0;
    double y = 0;
    Eigen::MatrixXd predictions(positions.rows(), 2);
    for (uint i = 0; i < positions.rows(); ++i) {
        using distance_func_t
            = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
        features::InterIndividualDistance<distance_func_t> iid;
        iid(e_positions.row(i), timestep);
        Eigen::MatrixXd ind_pos_t = e_positions.block(i, e_positions.cols() - 2, 1, 2);
        polygons::Point p(ind_pos_t(0), ind_pos_t(1));
        Eigen::MatrixXd dist_to_walls(1, 2);
        polygons::CircularCorridor<Params> cc;
        dist_to_walls << cc.distance_to_inner_wall(p), cc.distance_to_outer_wall(p);

        Eigen::MatrixXd sample(1, e_positions.cols() + velocities.cols() + 1 + 2);
        sample << positions.row(i),
            velocities.row(i),
            iid.get().block(0, 0, 1, 1),
            dist_to_walls,
            ind_pos_t;

        Eigen::MatrixXd pred;
#ifdef USE_ORIGINAL_LABELS
        uint label_idx = static_cast<int>(i / aggregate_window);
#else
        uint label_idx;
        if (i + 1 == positions.rows())
            label_idx = static_cast<int>(i / aggregate_window);
        else {
            Eigen::MatrixXd pos_block;
            if (i > 0)
                pos_block = e_positions.block(i - 1, 0, 2, e_positions.cols());
            else
                pos_block = e_positions.block(i, 0, 2, e_positions.cols());

            features::Alignment align;
            align(pos_block, timestep);

            Eigen::MatrixXd features(1, 2);
            uint row_idx;
            (i == 0) ? row_idx = 0 : row_idx = align.get().rows() - 2;
            features << iid.get()(0), align.get()(row_idx);
            label_idx = km.predict(features)(0);
        }
#endif
        pred = nn[labels(label_idx)].forward(sample.transpose()).transpose();

        if (i < 1) {
            x = e_positions(0, e_positions.cols() - 2);
            y = e_positions(0, e_positions.cols() - 1);
        }
        else {
            double new_x = x + pred(0) * timestep;
            double new_y = y + pred(1) * timestep;
            polygons::Point p(new_x, new_y);
            bool valid = cc.in_polygon(p);
            if (valid) {
                x = new_x;
                y = new_y;
            }
            else {
                // TODO: this doesn't work duh
                // double phi = std::fmod(std::atan2(y, x) + 2 * M_PI, 2 * M_PI);
                // double radius = std::sqrt(std::pow(x - cc.center().x(), 2.) + std::pow(y - cc.center().y(), 2.));
                // limbo::tools::rgen_double_t rgen(0, 1);
                // radius = rgen.rand() / 10 + cc.inner_radius();
                // x = radius * std::cos(phi) + cc.center().x();
                // y = radius * std::sin(phi) + cc.center().y();
            }
        }

        predictions(i, 0) = x;
        predictions(i, 1) = y;

        if (i + 1 == positions.rows())
            continue;
        e_positions(i + 1, e_positions.cols() - 2) = predictions(i, 0);
        e_positions(i + 1, e_positions.cols() - 1) = predictions(i, 1);
        e_velocities(i + 1, e_velocities.cols() - 2) = (e_positions(i + 1, e_positions.cols() - 2)
                                                           - e_positions(i, e_positions.cols() - 2))
            / timestep;
        e_velocities(i + 1, e_velocities.cols() - 1) = (e_positions(i + 1, e_positions.cols() - 1)
                                                           - e_positions(i, e_positions.cols() - 1))
            / timestep;
    }

    for (uint i = 0; i < e_positions.cols() / 2; ++i) {
        Eigen::MatrixXd bl = e_velocities.block(0, i * 2, e_velocities.rows(), 2).array().abs();
        std::cout << bl.rowwise().norm().mean() << " ";
    }
    std::cout << std::endl;

    Eigen::MatrixXd rolled = tools::rollMatrix(predictions, -1);
    rolled.row(rolled.rows() - 1) = removed.row(rolled.rows() - 1);
    double error = (removed - rolled).rowwise().norm().mean();

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
        ofs << error / 4 << std::endl;
    }

    archive.save(predictions,
        path + "/seg_" + std::to_string(exp_num) + "_virtual_traj_ex_" + std::to_string(exclude_idx));

    Eigen::MatrixXd extended_traj(positions.rows(), positions.cols() + rolled.cols());
    extended_traj << positions, rolled;
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