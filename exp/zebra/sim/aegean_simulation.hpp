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
                sim_time = 28800;
                stats_enabled = false; // don't want stats by default
            }

            int num_fish = 0;
            int num_robot = 3;
        };

        using IndividualPtr = std::shared_ptr<AegeanIndividual>;

        class AegeanSimulation : public Simulation {
        public:
            AegeanSimulation(simple_nn::NeuralNet network);

            void spin_once() override;

            std::vector<IndividualPtr> individuals() const;
            std::vector<IndividualPtr>& individuals();

        protected:
            void _init();

            AegeanSimSettings _aegean_sim_settings;
            std::vector<IndividualPtr> _individuals;
            simple_nn::NeuralNet _network;
            int _num_agents;
        };

        using AegeanSimulationPtr = std::shared_ptr<AegeanSimulation>;

    } // namespace simulation
} // namespace simu

#endif
