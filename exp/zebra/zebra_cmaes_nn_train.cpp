#include "nn_structure.hpp"

#include <tools/polygons/circular_corridor.hpp>
#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>

#include <histogram/histogram.hpp>
#include <histogram/hellinger_distance.hpp>

#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

#include "sim/aegean_simulation.hpp"
#include "sim/aegean_individual.hpp"
#include "sim/invalid_prediction_desc.hpp"

#include <limbo/opt/cmaes.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

#include <mutex>

using namespace aegean;
using namespace tools;
using namespace histogram;
using namespace features;

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };

    struct opt_cmaes : public limbo::defaults::opt_cmaes {
        BO_PARAM(int, max_fun_evals, 60000);
        /// 0 -> no elitism
        /// 1 -> elitism: reinjects the best-ever seen solution
        /// 2 -> initial elitism: reinject x0 as long as it is not improved upon
        /// 3 -> initial elitism on restart: restart if best encountered solution is not the the final
        BO_PARAM(int, elitism, 1);
    };
};

Histogram lvh(std::make_pair(0., 1.), 0.02);
Histogram ah(std::make_pair(0., 1.), 0.05);
Histogram iidh(std::make_pair(0., 360.), 5.);
Histogram nwh(std::make_pair(0., .05), .002);

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

struct ZebraSim {
public:
    ZebraSim(const std::vector<Eigen::MatrixXd>& orig_dists, int num_individuals, int num_segments, const std::string& path)
        : _archive(false),
          _orig_dists(orig_dists),
          _num_individuals(num_individuals),
          _num_segments(num_segments),
          _path(path),
          _iteration(0)
    {
        {
            std::ifstream ifs;
            ifs.open(_path + "/num_behaviours.dat");
            ifs >> _num_behaviours;
            ifs.close();
        }

        {
            std::ifstream ifs;
            ifs.open(_path + "/num_centroids.dat");
            ifs >> _num_centroids;
            ifs.close();
        }

        {
            std::ifstream ifs;
            ifs.open(_path + "/fps.dat");
            ifs >> _fps;
            ifs.close();
        }

        {
            std::ifstream ifs;
            ifs.open(_path + "/window_in_seconds.dat");
            ifs >> _window_in_seconds;
            ifs.close();
        }

        _aggregate_window = _window_in_seconds * (_fps / _num_centroids);
        _timestep = static_cast<float>(_num_centroids) / _fps;
    }

    limbo::opt::eval_t operator()(const Eigen::VectorXd& params, bool grad = false) const
    {
        {
            std::lock_guard<std::mutex> lock(_mtx);
            std::cout << "CMAES iteration: " << _iteration++ << std::endl;
        }

        simu::simulation::NNVec nn;
        nn = simu::simulation::NNVec(_num_behaviours);
        nn_strucutre::init_nn(nn, _num_behaviours, _num_individuals);

        Alignment align;
        polygons::CircularCorridor<Params> cc;
        using distance_func_t = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
        InterIndividualDistance<distance_func_t> iid;

        Eigen::MatrixXd cluster_centers;
        _archive.load(cluster_centers, _path + "/centroids_kmeans.dat");
        aegean::clustering::KMeans<> km(cluster_centers);

        for (uint b = 0; b < _num_behaviours; ++b) {
            Eigen::MatrixXd behaviour_specific_params;
            behaviour_specific_params = params.block(
                static_cast<int>(params.rows() / _num_behaviours) * b, 0,
                static_cast<int>(params.rows() / _num_behaviours), params.cols());
            nn[b]->set_weights(behaviour_specific_params);
        } // b

        double fit = 0;
        for (uint exp_num = 0; exp_num < _num_segments; ++exp_num) {
            // std::cout << "\t Experiment: " << exp_num << std::endl;

            Eigen::MatrixXd positions, velocities;
            _archive.load(positions, _path + "/seg_" + std::to_string(exp_num) + "_reconstructed_positions.dat");
            _archive.load(velocities, _path + "/seg_" + std::to_string(exp_num) + "_reconstructed_velocities.dat");

            for (uint exclude_idx = 0; exclude_idx < _num_individuals; ++exclude_idx) {
                // std::cout << "\t\t Individual: " << exclude_idx << std::endl;

                // distribution vectors
                Eigen::MatrixXd lin_vel_hist = Eigen::MatrixXd::Zero(1, 50);
                Eigen::MatrixXd align_hist = Eigen::MatrixXd::Zero(1, 20);
                Eigen::MatrixXd iid_hist = Eigen::MatrixXd::Zero(1, 72);
                Eigen::MatrixXd nearest_wall_hist = Eigen::MatrixXd::Zero(1, 25);

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
                    {static_cast<int>(exclude_idx)});
                sim.add_desc(std::make_shared<simu::desc::InvalidPrediction>(true));
                sim.aegean_sim_settings().aggregate_window = _aggregate_window;
                sim.aegean_sim_settings().timestep = _timestep;
                sim.sim_time() = positions.rows();
                sim.spin();

                {
                    Eigen::MatrixXd res_velocities(velocities.rows(), velocities.cols() / 2);
                    for (uint j = 0; j < res_velocities.cols(); ++j) {
                        res_velocities.col(j) = ((*generated_velocities).col(j * 2).array().pow(2) + (*generated_velocities).col(j * 2 + 1).array().pow(2)).array().sqrt();
                    }
                    Eigen::MatrixXd dist_to_nearest_wall(positions.rows(), positions.cols() / 2);
                    for (uint j = 0; j < dist_to_nearest_wall.cols(); ++j) {
                        for (uint r = 0; r < (*generated_positions).rows(); ++r) {
                            primitives::Point p((*generated_positions)(r, j * 2), (*generated_positions)(r, j * 2 + 1));
                            dist_to_nearest_wall(r, j) = cc.min_distance(p);
                        }
                    }
                    align((*generated_positions), _timestep);
                    iid((*generated_positions), _timestep);

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
                std::vector<Eigen::MatrixXd> dists{lin_vel_hist, align_hist, iid_hist, nearest_wall_hist};

                double invalid_perc = sim.descriptors()[0]->get()[0];

                double distances = 1;
                // std::cout << "\t\t";
                for (uint i = 0; i < dists.size(); ++i) {
                    double single_hd = 1 - _hd(_orig_dists[i], dists[i]);
                    distances *= single_hd;
                    // std::cout << single_hd << " ";
                }
                // std::cout << std::endl;
                fit += (1 - invalid_perc) * pow(distances, 1. / dists.size());
            } // exclude_idx

        } // exp_num

        fit /= _num_segments * _num_individuals;
        // std::cout << "\t Fitness: " << fit << std::endl;
        _archive.save(params, _path + "/cmaes_iter_" + std::to_string(_iteration) + "_weights_" + std::to_string(fit));

        return limbo::opt::no_grad(fit);
    }

private:
    mutable std::mutex _mtx;
    Archive _archive;
    std::vector<Eigen::MatrixXd> _orig_dists;
    uint _num_individuals;
    uint _num_segments;
    std::string _path;
    mutable int _iteration;
    uint _num_behaviours;
    uint _num_centroids;
    uint _fps;
    uint _window_in_seconds;
    uint _aggregate_window;
    float _timestep;
    HellingerDistance _hd;
};

int main(int argc, char** argv)
{
    assert(argc == 3);
    std::string path(argv[1]);
    const int num_segments = std::stoi(argv[2]);
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

    uint num_individuals;
    {
        Eigen::MatrixXd positions;
        archive.load(positions, path + "/seg_0_reconstructed_positions.dat");
        num_individuals = static_cast<uint>(positions.cols() / 2);
    }

    float timestep = static_cast<float>(centroids) / fps;

    std::vector<Eigen::MatrixXd> orig_dists = load_original_distributions(path, num_segments, timestep);
    ZebraSim sim(orig_dists, num_individuals, num_segments, std::string(path));
    limbo::opt::Cmaes<Params> cmaes;

    Eigen::MatrixXd random_params = Eigen::MatrixXd::Random(1772 * behaviours, 1);
    Eigen::VectorXd res_cmaes = cmaes(sim, random_params, false);

    std::cout << "Result with CMA-ES:\t" << res_cmaes.transpose()
              << " -> " << sim(res_cmaes).first << std::endl;
    archive.save(res_cmaes, path + "/cmaes_opt_weights");

    return 0;
}