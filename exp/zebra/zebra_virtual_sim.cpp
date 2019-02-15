#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <clustering/kmeans.hpp>
#include <tools/mathtools.hpp>

#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

#include <simple_nn/neural_net.hpp>

#include <Eigen/Core>
#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <ctime>

using namespace aegean;
using namespace tools;
using namespace clustering;

#define VEL_THRESHOLD (7. * M_PI / 180.)

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };

    struct sim {
        static constexpr uint timesteps = 28800;
        static constexpr uint num_individuals = 6;
    };
};

int main(int argc, char** argv)
{
    std::srand(std::time(NULL));

    assert(argc == 2);
    std::string path(argv[1]);
    Archive archive(false);

    // load basic parameters concerning the data set for this experiment
    uint behaviours;
    {
        std::ifstream ifs;
        ifs.open(path + "/num_behaviours.dat");
        ifs >> behaviours;
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

    Eigen::MatrixXd centroids;
    archive.load(centroids, path + "/centroids_kmeans.dat");
    KMeans<> km(centroids);

    float timestep = static_cast<float>(num_centroids) / fps;
    uint num_timesteps = Params::sim::timesteps;
    uint num_individuals = Params::sim::num_individuals;

    // position and velocity matrices
    Eigen::MatrixXd positions(num_timesteps, num_individuals * 2);
    Eigen::MatrixXd velocities(num_timesteps, num_individuals * 2);
    Eigen::MatrixXd radius(1, num_individuals);
    Eigen::MatrixXd phi(1, num_individuals);

    // nn structure
    // TODO: the nn structure needs to be updated every time. Change this!
    std::vector<simple_nn::NeuralNet> nn(behaviours);
    for (uint b = 0; b < behaviours; ++b) {
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(
            positions.cols() - 2 + velocities.cols() - 2, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(100, 100);
        nn[b].add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(100, 3); // pos_in_cc phi.cos phi.sin
        Eigen::MatrixXd weights;
        archive.load(weights, path + "/nn_controller_weights_" + std::to_string(b) + ".dat");
        nn[b].set_weights(weights);
    }
    limbo::tools::rgen_double_t rgend(0., 1.);
    limbo::tools::rgen_int_t rgeni(0, nn.size() - 1);

    // bootstrap data by randomly assigning
    {
        polygons::CircularCorridor<Params> cc;
        uint i = 0;
        while (i < num_individuals) {
            Eigen::VectorXd rnd = Eigen::VectorXd::Random(2);
            primitives::Point pt(rnd(0), rnd(1));
            if (cc.is_valid(pt)) {
                positions(0, i * 2) = pt.x();
                positions(0, i * 2 + 1) = pt.y();
                double x = pt.x() - cc.center().x();
                double y = pt.y() - cc.center().y();
                radius(i) = std::sqrt(pow(x, 2) + pow(y, 2));
                phi(i) = std::fmod(std::atan2(y, x) + 2 * M_PI, 2 * M_PI);
                ++i;
            }
        }
        Eigen::VectorXd noise = 0.3 * (Eigen::VectorXd::Random(num_individuals * 2).array() - 0.7) + 1;
        positions.row(1) = positions.row(0).array() * noise.transpose().array();
        velocities.row(0) = (positions.row(1) - positions.row(0)) / timestep;
    }

    // spin virtual sim
    std::vector<uint> ind_idcs(num_individuals);
    std::generate(ind_idcs.begin(), ind_idcs.end(), []() { 
        static uint idx = -1; ++idx; return idx; });
    for (uint i = 0; i < num_timesteps - 1; ++i) {
        using distance_func_t
            = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
        features::InterIndividualDistance<distance_func_t> iid;
        features::Alignment align;
        iid(positions.row(i), timestep);

        Eigen::MatrixXd pos_block;
        if (i > 0)
            pos_block = positions.block(i - 1, 0, 2, positions.cols());
        else
            pos_block = positions.block(i, 0, 2, positions.cols());
        align(pos_block, timestep);

        Eigen::MatrixXd sample(1, 2);
        sample << iid.get()(0), align.get()(0);
        uint label = km.predict(sample)(0);

        std::vector<uint> shuffled_idcs = ind_idcs;
        std::random_shuffle(shuffled_idcs.begin(), shuffled_idcs.end());
        Eigen::VectorXi controller_idcs = Eigen::VectorXi::Zero(num_individuals);
        for (const uint focal_idx : shuffled_idcs) {
            Eigen::MatrixXd input((num_individuals - 1) * 4, 1); // x y vx vy for n-1 individuals
            uint cur_idx = 0;
            for (const uint idx : ind_idcs) {
                if (idx != focal_idx) {
                    input(cur_idx * 2) = positions(i, idx * 2);
                    input(cur_idx * 2 + 1) = positions(i, idx * 2 + 1);
                    input(cur_idx * 2 + (num_individuals - 1) * 2) = velocities(i, idx * 2);
                    input(cur_idx * 2 + 1 + (num_individuals - 1) * 2) = velocities(i, idx * 2 + 1);
                    ++cur_idx;
                }
            }

            // TODO: introduce ethogram transition probabilities
            // if (i % aggregate_window == 0) {
            if (rgend.rand() > 0.84)
                controller_idcs(focal_idx) = rgeni.rand();
            else
                controller_idcs(focal_idx) = label;
            // }
            Eigen::MatrixXd pred = nn[controller_idcs(focal_idx)].forward(input);

            polygons::CircularCorridor<Params> cc;
            double dradius = pred(0) * timestep;
            double dphi = std::atan2(pred(2) * timestep, pred(1) * timestep);
            double thres = 22 * M_PI / 180. * timestep;
            double diff = phi(focal_idx) + dphi;
            if (phi(focal_idx) + dphi < 0)
                diff = M_PI + phi(focal_idx) + dphi;
            else if (phi(focal_idx) + dphi > 2 * M_PI)
                diff = phi(focal_idx) + dphi - M_PI;
            if (abs(diff) > thres)
                diff = sgn(dphi) * thres;
            phi(focal_idx) += diff;
            radius(focal_idx) += pred(0) * timestep;
            if (radius(focal_idx) > cc.outer_radius())
                radius(focal_idx) = cc.inner_radius() + 0.05;

            if (radius(focal_idx) < cc.inner_radius())
                radius(focal_idx) = cc.inner_radius() + 0.05;

            double x = radius(focal_idx) * std::cos(phi(focal_idx)) + cc.center().x();
            double y = radius(focal_idx) * std::sin(phi(focal_idx)) + cc.center().y();
            positions(i + 1, focal_idx * 2) = x;
            positions(i + 1, focal_idx * 2 + 1) = y;
            velocities(i + 1, focal_idx * 2) = (positions(i + 1, focal_idx * 2) - positions(i, focal_idx * 2))
                / timestep;
            velocities(i + 1, focal_idx * 2 + 1) = (positions(i + 1, focal_idx * 2 + 1) - positions(i, focal_idx * 2 + 1))
                / timestep;
        }
    }

    Eigen::MatrixXd vel = (positions - rollMatrix(positions, -1)) / timestep;
    Eigen::MatrixXd vel_test = vel.block(0, 0, vel.rows() - 1, positions.cols());
    std::cout << vel_test.colwise().maxCoeff() << std::endl;

    archive.save(positions, "test");

    return 0;
}