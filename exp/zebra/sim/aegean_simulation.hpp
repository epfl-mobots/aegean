#ifndef SIMU_SIMULATION_AEGEAN_SIMULATION_HPP
#define SIMU_SIMULATION_AEGEAN_SIMULATION_HPP

#include "aegean_individual.hpp"
#include <simulation/simulation.hpp>

#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>
#include <limbo/opt/adam.hpp>
#include <clustering/kmeans.hpp>

#include <Eigen/Core>

namespace simu {
    namespace simulation {

        struct AegeanSimSettings : public Settings {
            AegeanSimSettings()
            {
                sim_time = 25000;
                stats_enabled = false; // don't want stats by default
                timestep = 0.0666666666667;
            }

            int num_agents = -1;
            int num_fish = -1;
            int num_robot = -1;
            int aggregate_window = -1;
        };

        using NNVec = std::vector<std::shared_ptr<simple_nn::NeuralNet>>;
        using namespace aegean;
        using namespace clustering;

        class AegeanSimulation : public Simulation {
        public:
            AegeanSimulation(NNVec,
                std::shared_ptr<KMeans<>> kmeans,
                std::shared_ptr<Eigen::MatrixXd> positions,
                std::shared_ptr<Eigen::MatrixXd> velocities,
                std::shared_ptr<Eigen::MatrixXd> generated_positions, const std::vector<int>& robot_idcs = {},
                const Eigen::MatrixXd& labels = Eigen::MatrixXd());

            void spin_once() override;

            const NNVec network() const;

            const std::shared_ptr<const Eigen::MatrixXd> orig_positions() const;
            std::shared_ptr<Eigen::MatrixXd> orig_positions();

            const std::shared_ptr<const Eigen::MatrixXd> orig_velocities() const;
            std::shared_ptr<Eigen::MatrixXd> orig_velocities();

            const std::shared_ptr<const Eigen::MatrixXd> generated_positions() const;
            std::shared_ptr<Eigen::MatrixXd> generated_positions();

            std::vector<IndividualPtr> individuals() const;
            std::vector<IndividualPtr>& individuals();

            AegeanSimSettings aegean_sim_settings() const;
            AegeanSimSettings& aegean_sim_settings();

            bool has_labels() const;
            const Eigen::MatrixXd& labels() const;

            const std::shared_ptr<KMeans<>> kmeans() const;

        protected:
            void _init();

            AegeanSimSettings _aegean_sim_settings;
            std::vector<IndividualPtr> _individuals;

            NNVec _network;
            std::shared_ptr<KMeans<>> _kmeans;
            std::shared_ptr<Eigen::MatrixXd> _positions;
            std::shared_ptr<Eigen::MatrixXd> _velocities;
            std::shared_ptr<Eigen::MatrixXd> _generated_positions;

            std::vector<int> _robot_idcs;
            Eigen::MatrixXd _labels;
        };

        using AegeanSimulationPtr = std::shared_ptr<AegeanSimulation>;

    } // namespace simulation
} // namespace simu

#endif
