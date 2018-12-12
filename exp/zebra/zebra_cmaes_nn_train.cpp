#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>

#include <histogram/histogram.hpp>
#include <histogram/hellinger_distance.hpp>

#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

#include <limbo/opt/cmaes.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace aegean;
using namespace tools;
using namespace histogram;
using namespace features;

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };

    struct opt_cmaes : public limbo::defaults::opt_cmaes {
    };
};

Histogram lvh(std::make_pair(0., 1.), 0.02);
Histogram ah(std::make_pair(0., 1.), 0.05);
Histogram iidh(std::make_pair(0., 360.), 5.);
Histogram nwh(std::make_pair(0., .05), .002);

limbo::opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad = false)
{
    double v = -(params.array() - 0.5).square().sum();
    if (!eval_grad)
        return limbo::opt::no_grad(v);
    Eigen::VectorXd grad = (-2 * params).array() + 1.0;
    return {v, grad};
}

std::vector<Eigen::MatrixXd> load_original_distributions(const std::string& path, const int num_segments, const float timestep)
{
    Archive archive(false);
    Alignment align;
    polygons::CircularCorridor<Params> cc;
    using distance_func_t = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    InterIndividualDistance<distance_func_t> iid;

    Eigen::MatrixXd lin_vel_hist = Eigen::MatrixXd::Zero(1, 50);
    Eigen::MatrixXd align_hist = Eigen::MatrixXd::Zero(1, 20);
    Eigen::MatrixXd iid_hist = Eigen::MatrixXd::Zero(1, 72);
    Eigen::MatrixXd nearest_wall_hist = Eigen::MatrixXd::Zero(1, 25);

    // for every available experiment segment we compute the distributions
    for (int i = 0; i < num_segments; ++i) {
        Eigen::MatrixXd positions, velocities;
        archive.load(positions, path + "/seg_" + std::to_string(i) + "_reconstructed_positions.dat", 0);
        archive.load(velocities, path + "/seg_" + std::to_string(i) + "_reconstructed_velocities.dat", 0);

        Eigen::MatrixXd res_velocities(velocities.rows(), velocities.cols() / 2);
        for (uint j = 0; j < res_velocities.cols(); ++j) {
            res_velocities.col(j) = (velocities.col(j * 2).array().pow(2) + velocities.col(j * 2 + 1).array().pow(2)).array().sqrt();
        }
        Eigen::MatrixXd dist_to_nearest_wall(positions.rows(), positions.cols() / 2);
        for (uint j = 0; j < dist_to_nearest_wall.cols(); ++j) {
            for (uint r = 0; r < positions.rows(); ++r) {
                primitives::Point p(positions(r, j * 2), positions(r, j * 2 + 1));
                dist_to_nearest_wall(r, j) = cc.min_distance(p);
            }
        }
        align(positions, timestep);
        iid(positions, timestep);

        // aggregate histograms
        lin_vel_hist += lvh(res_velocities);
        align_hist += ah(align.get());
        iid_hist += iidh(iid.get() * 360);
        nearest_wall_hist += nwh(dist_to_nearest_wall.rowwise().mean());
    }
    // normalize into probs
    lin_vel_hist /= lin_vel_hist.sum();
    align_hist /= align_hist.sum();
    iid_hist /= iid_hist.sum();
    nearest_wall_hist /= nearest_wall_hist.sum();

    archive.save(lin_vel_hist, path + "/linear_velocity_distribution");
    archive.save(align_hist, path + "/alignment_distribution");
    archive.save(iid_hist, path + "/inter_individual_distribution");
    archive.save(nearest_wall_hist, path + "/nearest_wall_distribution");

    std::vector<Eigen::MatrixXd> dists{lin_vel_hist, align_hist, iid_hist, nearest_wall_hist};
    return dists;
}

int main(int argc, char** argv)
{
    assert(argc == 3);
    std::string path(argv[1]);
    const int num_segments = std::stoi(argv[2]);

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

    float timestep = static_cast<float>(centroids) / fps;

    std::vector<Eigen::MatrixXd> orig_dists = load_original_distributions(path, num_segments, timestep);

    // limbo::opt::Cmaes<Params> cmaes;
    // Eigen::VectorXd res_cmaes = cmaes(my_function, limbo::tools::random_vector(2), false);

    // std::cout << "Result with CMA-ES:\t" << res_cmaes.transpose()
    //           << " -> " << my_function(res_cmaes).first << std::endl;

    return 0;
}