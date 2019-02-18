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
        BO_PARAM(int, elitism, 0);
    };
};

struct Histograms {
public:
    Histograms()
        : linear_velocity(std::make_pair(0., .3), 0.03),
          alignment(std::make_pair(0., 1.), 0.05),
          inter_individual(std::make_pair(0., 180.), 5.),
          outer_wall_distance(std::make_pair(0., .1), .002)
    {
    }

    std::vector<Histogram> to_vec() const
    {
        return std::vector<Histogram>{
            linear_velocity,
            alignment,
            inter_individual,
            outer_wall_distance};
    }

    Histogram linear_velocity;
    Histogram alignment;
    Histogram inter_individual;
    Histogram outer_wall_distance;
};

struct Distributions {
public:
    Distributions(const Histograms& hists, std::vector<Eigen::MatrixXd> data)
    {
        std::vector<Histogram> hist_vec = hists.to_vec();
        assert(hist_vec.size() == data.size());

        for (uint i = 0; i < hist_vec.size(); ++i) {
            Eigen::MatrixXd d = hist_vec[i](data[i]);
            d /= d.sum(); // in this step we normalize the histogram into probabilities
            _dists.push_back(d);
        }
    }

    double fit(const Distributions& original)
    {
        std::vector<Eigen::MatrixXd> original_dists = original.distributions();
        assert(original_dists.size() == _dists.size());
        _individual_fitness.clear();

        double distances = 1;
        for (uint i = 0; i < _dists.size(); ++i) {
            double single_hd = 1 - _hd(original_dists[i], _dists[i]);
            _individual_fitness.push_back(single_hd);
            distances *= single_hd;
        }
        _aggr_fit = pow(distances, 1. / _dists.size());
        return _aggr_fit;
    }

    friend std::ostream& operator<<(std::ostream& os, const Distributions& dt)
    {
        os << dt.aggregated_fitness() << " | ";
        for (uint i = 0; i < dt.individual_fitness().size(); ++i)
            os << dt.individual_fitness()[i] << " ";
        return os;
    }

    double aggregated_fitness() const { return _aggr_fit; }
    const std::vector<double>& individual_fitness() const { return _individual_fitness; }
    const std::vector<Eigen::MatrixXd>& distributions() const { return _dists; }

protected:
    std::vector<Eigen::MatrixXd> _dists;
    HellingerDistance _hd;
    std::vector<double> _individual_fitness;
    double _aggr_fit;
};

const std::vector<Distributions> load_original_distributions(const std::string& path, const int num_segments, const float timestep)
{
    Archive archive(false);
    Alignment align;
    polygons::CircularCorridor<Params> cc;
    using distance_func_t = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    InterIndividualDistance<distance_func_t> iid;
    Histograms hists;
    std::vector<Distributions> dists;

    // for every available experiment segment we compute the distributions
    for (int i = 0; i < num_segments; ++i) {
        Eigen::MatrixXd positions, velocities;
        archive.load(positions, path + "/seg_" + std::to_string(i) + "_reconstructed_positions.dat", 0);
        archive.load(velocities, path + "/seg_" + std::to_string(i) + "_reconstructed_velocities.dat", 0);

        Eigen::MatrixXd res_velocities(velocities.rows(), velocities.cols() / 2);
        for (uint r = 0; r < res_velocities.rows(); ++r) {
            for (uint c = 0; c < res_velocities.cols(); ++c) {
                double phi = std::atan2(velocities(r, c * 2 + 1), velocities(r, c * 2));
                double resultant = std::sqrt(
                    std::pow(velocities(r, c * 2), 2)
                    + std::pow(velocities(r, c * 2 + 1), 2)
                    + 2 * velocities(r, c * 2) * velocities(r, c * 2)
                        * std::sin(phi));
                res_velocities(r, c) = resultant;
            }
        }

        Eigen::MatrixXd dist_to_outer_wall(positions.rows(), positions.cols() / 2);
        for (uint j = 0; j < dist_to_outer_wall.cols(); ++j) {
            for (uint r = 0; r < positions.rows(); ++r) {
                primitives::Point p(positions(r, j * 2), positions(r, j * 2 + 1));
                dist_to_outer_wall(r, j) = cc.distance_to_outer_wall(p);
            }
        }
        align(positions, timestep);
        iid(positions, timestep);

        std::vector<Eigen::MatrixXd> data = {
            res_velocities,
            align.get(),
            iid.get() * 360,
            dist_to_outer_wall};
        dists.push_back(Distributions(hists, data));
    }

    return dists;
}

struct ZebraSim {
public:
    ZebraSim(const std::vector<Distributions>& orig_dists, int num_individuals, int num_segments, const std::string& path)
        : _archive(false),
          _orig_dists(orig_dists),
          _num_individuals(num_individuals),
          _num_segments(num_segments),
          _path(path),
          _evaluation(0)
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

        _fit_file.open(_path + "/fitness.dat"); // erase last runs
        _fit_file.close();
    }

    limbo::opt::eval_t operator()(const Eigen::VectorXd& params, bool grad = false) const
    {
        int current_eval;
        {
            std::lock_guard<std::mutex> lock(_mtx);
            current_eval = _evaluation;
            std::cout << "CMAES evaluation: " << _evaluation++ << std::endl;
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

        double fit = 0; // overall fitness
        double invalid_percentage = 0;

        for (uint exp_num = 0; exp_num < _num_segments; ++exp_num) {
            Eigen::MatrixXd positions, velocities;
            _archive.load(positions, _path + "/seg_" + std::to_string(exp_num) + "_reconstructed_positions.dat");
            _archive.load(velocities, _path + "/seg_" + std::to_string(exp_num) + "_reconstructed_velocities.dat");

            for (uint exclude_idx = 0; exclude_idx < _num_individuals; ++exclude_idx) {
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
                sim.sim_settings().timestep = _timestep;
                sim.sim_time() = positions.rows();
                sim.spin();

                {
                    Eigen::MatrixXd res_velocities(sim.sim_time(), 1);
                    for (uint r = 0; r < res_velocities.rows(); ++r) {
                        double phi = std::atan2((*generated_velocities)(r, exclude_idx * 2 + 1), (*generated_velocities)(r, exclude_idx * 2));
                        double resultant = std::sqrt(
                            std::pow((*generated_velocities)(r, exclude_idx * 2), 2)
                            + std::pow((*generated_velocities)(r, exclude_idx * 2 + 1), 2)
                            + 2 * (*generated_velocities)(r, exclude_idx * 2) * (*generated_velocities)(r, exclude_idx * 2)
                                * std::sin(phi));
                        res_velocities(r) = resultant;
                    }

                    Eigen::MatrixXd dist_to_outer_wall(sim.sim_time(), 1);
                    for (uint r = 0; r < dist_to_outer_wall.rows(); ++r) {
                        primitives::Point p((*generated_positions)(r, exclude_idx * 2), (*generated_positions)(r, exclude_idx * 2 + 1));
                        dist_to_outer_wall(r, 0) = cc.distance_to_outer_wall(p);
                    }
                    align((*generated_positions).block(0, 0, sim.sim_time(), positions.cols()), _timestep);
                    iid((*generated_positions).block(0, 0, sim.sim_time(), positions.cols()), _timestep);

                    std::vector<Eigen::MatrixXd> data = {
                        res_velocities,
                        align.get(),
                        iid.get() * 360,
                        dist_to_outer_wall};
                    Distributions dist(_hists, data);
                    invalid_percentage += sim.descriptors()[0]->get()[0];
                    fit += dist.fit(_orig_dists[exp_num]);
                    std::cout << dist << std::endl;
                }
            } // exclude_idx
        } // exp_num

        fit /= _num_individuals * _num_segments;
        invalid_percentage /= _num_individuals * _num_segments;
        std::cout << "Aggregated reward: " << fit << std::endl;
        std::cout << "Motionless percentage: " << invalid_percentage << std::endl;
        _archive.save(params, _path + "/cmaes_weights_e_" + std::to_string(current_eval));
        {
            std::lock_guard<std::mutex> lock(_mtx);
            _fit_file.open(_path + "/fitness.dat", std::ios_base::app);
            _fit_file << current_eval << " " << fit << " " << invalid_percentage << "\n";
            _fit_file.close();
        }

        return limbo::opt::no_grad(fit);
    }

private:
    mutable std::mutex _mtx;
    Archive _archive;
    std::vector<Distributions> _orig_dists;
    uint _num_individuals;
    uint _num_segments;
    std::string _path;
    mutable int _evaluation;
    uint _num_behaviours;
    uint _num_centroids;
    uint _fps;
    uint _window_in_seconds;
    uint _aggregate_window;
    float _timestep;
    Histograms _hists;
    mutable std::ofstream _fit_file;
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

    std::vector<Distributions> orig_dists = load_original_distributions(path, num_segments, timestep);
    ZebraSim sim(orig_dists, num_individuals, num_segments, std::string(path));
    limbo::opt::Cmaes<Params> cmaes;

    Eigen::MatrixXd random_params = Eigen::MatrixXd::Random(982 * behaviours, 1);
    Eigen::VectorXd res_cmaes = cmaes(sim, random_params, false);

    std::cout << "Result with CMA-ES:\t" << res_cmaes.transpose()
              << " -> " << sim(res_cmaes).first << std::endl;
    archive.save(res_cmaes, path + "/cmaes_opt_weights");

    return 0;
}
