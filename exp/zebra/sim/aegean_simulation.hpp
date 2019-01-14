#ifndef SIMU_SIMULATION_AEGEAN_SIMULATION_HPP
#define SIMU_SIMULATION_AEGEAN_SIMULATION_HPP

#include "aegean_individual.hpp"
#include <simulation/simulation.hpp>

#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>
#include <limbo/opt/adam.hpp>

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

            int num_agents = 3;
            int num_fish = -1;
            int num_robot = -1;
        };

        class AegeanSimulation : public Simulation {
        public:
            AegeanSimulation(const simple_nn::NeuralNet& network, std::shared_ptr<Eigen::MatrixXd> positions, std::shared_ptr<Eigen::MatrixXd> velocities, const std::vector<int>& robot_idcs = {});

            void spin_once() override;

            const std::shared_ptr<const Eigen::MatrixXd> orig_positions() const;
            std::shared_ptr<Eigen::MatrixXd> orig_positions();
            const std::shared_ptr<const Eigen::MatrixXd> orig_velocities() const;
            std::shared_ptr<Eigen::MatrixXd> orig_velocities();
            std::vector<IndividualPtr> individuals() const;
            std::vector<IndividualPtr>& individuals();
            AegeanSimSettings aegean_sim_settings() const;
            AegeanSimSettings& aegean_sim_settings();

        protected:
            void _init();

            AegeanSimSettings _aegean_sim_settings;
            std::vector<IndividualPtr> _individuals;

            simple_nn::NeuralNet _network;
            std::shared_ptr<Eigen::MatrixXd> _positions;
            std::shared_ptr<Eigen::MatrixXd> _velocities;
            std::vector<int> _robot_idcs;
        };

        using AegeanSimulationPtr = std::shared_ptr<AegeanSimulation>;

    } // namespace simulation
} // namespace simu

#endif
