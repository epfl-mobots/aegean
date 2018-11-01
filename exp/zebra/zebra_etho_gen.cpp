#include <ethogram/automated_ethogram.hpp>
#include <clustering/kmeans.hpp>
#include <clustering/opt/gap_statistic.hpp>
#include <clustering/opt/no_opt.hpp>

#include <tools/archive.hpp>
#include <tools/experiment_data_frame.hpp>
#include <tools/mathtools.hpp>
#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>

#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>

using namespace aegean;
using namespace clustering;
using namespace tools;
using namespace opt;
using namespace ethogram;

struct Params {
    struct CircularCorridor : public defaults::CircularCorridor {
    };
};

int main(int argc, char** argv)
{
    assert(argc == 2);
    std::string path(argv[1]);

    // list of files to process
    std::vector<std::string> files = {
        path + "/fish_only/2_fish_only/trajectories.txt",
        path + "/fish_only/3_fish_only/trajectories.txt",
        path + "/fish_only/4_fish_only/trajectories.txt",
        path + "/fish_only/5_fish_only/trajectories.txt",
        path + "/fish_only/6_fish_only/trajectories.txt",
        path + "/fish_only/7_fish_only/trajectories.txt",
        path + "/fish_only/8_fish_only/trajectories.txt",
        path + "/fish_only/9_fish_only/trajectories.txt",
        path + "/fish_only/10_fish_only/trajectories.txt",
        path + "/fish_only/19_fish_only/trajectories.txt"};

    // setting the setup specifications (e.g., camera specs, setup scale, etc)
    int initial_keep = 27855;
    int process_frames = 27000;
    int fps = 15;
    int centroids = 3;
    double scale = 1.13 / 1024;
    uint window_in_seconds = 3;
    uint aggregate_frames = static_cast<int>(fps / centroids) * window_in_seconds;

    using distance_func_t
        = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;
    using reconstruction_t = reconstruction::CSpace<polygons::CircularCorridor<Params>>;
    using features_t = boost::fusion::vector<
        features::InterIndividualDistance<distance_func_t>,
        features::Alignment>;

    using ExpDataFrame = ExperimentDataFrame<reconfun<reconstruction_t>, featset<features_t>>;
    std::vector<ExpDataFrame> dataframes;
    Eigen::MatrixXd data;
    std::vector<std::string> feature_names;
    std::vector<uint> exp_segment_idcs(1, 0);
    Archive dl(false);
    for (const std::string f : files) {
        // first we load the raw trajectories as generated in idTracker
        Eigen::MatrixXd exp_data;
        dl.load<Eigen::MatrixXd>(exp_data, f, 1, '\t');
        std::vector<uint> prob_cols = {2, 5, 8, 11, 14, 17};
        removeCols(exp_data, prob_cols); // remove idTracker probability cols
        Eigen::MatrixXd positions = exp_data.block(exp_data.rows() - initial_keep, 0, initial_keep, exp_data.cols());

        // we process the raw positions to extract useful features for every timestep
        ExpDataFrame edf(positions, fps, centroids, scale, initial_keep - process_frames);
        Eigen::MatrixXd fm = edf.get_feature_matrix();
        dataframes.push_back(edf);

        Eigen::MatrixXd avg_fm;
        for (uint i = 0; i < fm.rows(); i += aggregate_frames) {
            avg_fm.conservativeResize(avg_fm.rows() + 1, fm.cols());
            avg_fm.row(avg_fm.rows() - 1) = fm.block(i, 0, aggregate_frames, fm.cols()).colwise().mean();
        }

        // append all features to one matrix to compute an complete ethogram
        if (data.size()) {
            Eigen::MatrixXd appendData(data.rows() + avg_fm.rows(), avg_fm.cols());
            appendData << data, avg_fm;
            data = appendData;
        }
        else {
            data = avg_fm;
            feature_names = edf.feature_names();
        }
        exp_segment_idcs.push_back(exp_segment_idcs[exp_segment_idcs.size() - 1] + avg_fm.rows());
    }

    using clustering_t = KMeans<aegean::defaults::KMeansPlusPlus>;
    // using clustering_opt_t = GapStatistic<clustering_t, 2, 10>;
    using clustering_opt_t = NoOpt<3>;

    AutomatedEthogram<
        clusteringmethod<clustering_t>,
        clusteringopt<clustering_opt_t>>
        etho(data, exp_segment_idcs);
    etho.compute();
    etho.save();

    std::cout << "Cluster centroids (KMeans)" << std::endl;
    std::cout << etho.model().centroids() << std::endl
              << std::endl;
    ;

    // saving
    for (uint i = 0; i < dataframes.size(); ++i) {
        etho.archive().save(dataframes[i].positions(),
            std::string("seg_") + std::to_string(i) + std::string("_reconstructed_positions"));

        etho.archive().save(dataframes[i].velocities(),
            std::string("seg_") + std::to_string(i) + std::string("_reconstructed_velocities"));

        Eigen::MatrixXd features = dataframes[i].get_feature_matrix();
        for (uint j = 0; j < features.cols(); ++j) {
            etho.archive().save(features.col(j),
                std::string("seg_") + std::to_string(i) + std::string("_") + feature_names[j]);
        }
    }

    {
        std::ofstream ofs(etho.etho_path() + "/num_centroids.dat");
        ofs << centroids << std::endl;
    }
    {
        std::ofstream ofs(etho.etho_path() + "/window_in_seconds.dat");
        ofs << window_in_seconds << std::endl;
    }
    {
        std::ofstream ofs(etho.etho_path() + "/num_behaviours.dat");
        ofs << etho.num_behaviours() << std::endl;
    }
    {
        std::ofstream ofs(etho.etho_path() + "/fps.dat");
        ofs << fps << std::endl;
    }

    return 0;
}