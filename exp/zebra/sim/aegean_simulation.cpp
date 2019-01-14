#include "aegean_simulation.hpp"

namespace simu {
    namespace simulation {
        AegeanSimulation::AegeanSimulation(simple_nn::NeuralNet network) : Simulation(false)
        {
            _init();
            Simulation::_init();
            _network = network;
        }

        void AegeanSimulation::spin_once()
        {
            // using namespace types;
            std::vector<int> idcs;
            for (int i = 0; i < _num_agents; ++i)
                idcs.push_back(i);
            std::random_device rd;
            std::mt19937 g(rd());

            // stimulate the fish to drive their movement decisions
            std::shuffle(idcs.begin(), idcs.end(), g);
            for (const int idx : idcs)
                _individuals[static_cast<size_t>(idx)]->stimulate(std::make_shared<AegeanSimulation>(*this));

            // apply intuitions and move accordingly
            std::shuffle(idcs.begin(), idcs.end(), g);
            for (const int idx : idcs)
                _individuals[static_cast<size_t>(idx)]->move(std::make_shared<AegeanSimulation>(*this));

            // update statistics
            _update_stats(std::make_shared<AegeanSimulation>(*this));
            _update_descriptors(std::make_shared<AegeanSimulation>(*this));

            Simulation::spin_once();
        }

        void AegeanSimulation::_init()
        {
            _sim_settings.sim_time = _aegean_sim_settings.sim_time;
            _sim_settings.stats_enabled = _aegean_sim_settings.stats_enabled;
            _num_agents = _aegean_sim_settings.num_fish + _aegean_sim_settings.num_robot;

            _individuals.resize(static_cast<size_t>(_num_agents));
            for (size_t i = 0; i < _individuals.size(); ++i) {
                _individuals[i] = std::make_shared<AegeanIndividual>();
                _individuals[i]->id() = static_cast<int>(i);
            }
            for (size_t i = 0; i < static_cast<size_t>(_aegean_sim_settings.num_robot); ++i)
                _individuals[i]->is_robot() = true;
        }

        std::vector<IndividualPtr> AegeanSimulation::individuals() const { return _individuals; }
        std::vector<IndividualPtr>& AegeanSimulation::individuals() { return _individuals; }
    } // namespace simulation
} // namespace simu