#include "aegean_simulation.hpp"

namespace simu {
    namespace simulation {
        AegeanSimulation::AegeanSimulation(NNVec network, std::shared_ptr<Eigen::MatrixXd> positions, std::shared_ptr<Eigen::MatrixXd> velocities, const std::vector<int>& robot_idcs, const std::vector<int>& labels)
            : Simulation(false),
              _positions(positions),
              _velocities(velocities),
              _robot_idcs(robot_idcs),
              _labels(labels)
        {
            assert(_positions->rows() > 0);
            assert(_velocities->rows() > 0);
            assert(_positions->rows() == _velocities->rows());
            assert(_positions->cols() == _velocities->cols());

            _init();
            Simulation::_init();
            _network = network;
        }

        void AegeanSimulation::spin_once()
        {
            // using namespace types;
            std::vector<int> idcs = {0, 1, 2};
            for (int i = 0; i < _aegean_sim_settings.num_agents; ++i)
                idcs.push_back(i);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(idcs.begin(), idcs.end(), g);

            // stimulate the fish to drive their movement decisions
            for (const int idx : idcs)
                _individuals[static_cast<size_t>(idx)]->stimulate(std::make_shared<AegeanSimulation>(*this));

            // apply intuitions and move accordingly
            for (const int idx : idcs)
                _individuals[static_cast<size_t>(idx)]->move(std::make_shared<AegeanSimulation>(*this));

            // update statistics
            _update_stats(std::make_shared<AegeanSimulation>(*this));
            _update_descriptors(std::make_shared<AegeanSimulation>(*this));

            Simulation::spin_once();
        }

        void AegeanSimulation::_init()
        {
            _aegean_sim_settings.num_robot = _robot_idcs.size();
            _aegean_sim_settings.num_fish = _aegean_sim_settings.num_agents - _robot_idcs.size();
            assert(_aegean_sim_settings.num_fish > 0);
            assert(_aegean_sim_settings.num_agents == _aegean_sim_settings.num_robot + _aegean_sim_settings.num_fish);
            if (_robot_idcs.size()) {
                assert(_aegean_sim_settings.num_agents >= *std::max_element(std::begin(_robot_idcs), std::end(_robot_idcs)));
            }
            _sim_settings.sim_time = _aegean_sim_settings.sim_time;
            _sim_settings.stats_enabled = _aegean_sim_settings.stats_enabled;

            _individuals.resize(static_cast<size_t>(_aegean_sim_settings.num_agents));
            for (size_t i = 0; i < _individuals.size(); ++i) {
                _individuals[i] = std::make_shared<AegeanIndividual>();
                _individuals[i]->id() = static_cast<int>(i);

                if (_robot_idcs.size() && std::find(_robot_idcs.begin(), _robot_idcs.end(), i) != _robot_idcs.end()) {
                    _individuals[i]->is_robot() = true;
                }
                else {
                    _individuals[i]->is_robot() = false;
                }
            }
        }

        const NNVec AegeanSimulation::network() const { return _network; }

        const std::shared_ptr<const Eigen::MatrixXd> AegeanSimulation::orig_positions() const { return _positions; }
        std::shared_ptr<Eigen::MatrixXd> AegeanSimulation::orig_positions() { return _positions; }

        const std::shared_ptr<const Eigen::MatrixXd> AegeanSimulation::orig_velocities() const { return _velocities; }
        std::shared_ptr<Eigen::MatrixXd> AegeanSimulation::orig_velocities() { return _velocities; }

        std::vector<IndividualPtr> AegeanSimulation::individuals() const { return _individuals; }
        std::vector<IndividualPtr>& AegeanSimulation::individuals() { return _individuals; }

        AegeanSimSettings AegeanSimulation::aegean_sim_settings() const { return _aegean_sim_settings; }
        AegeanSimSettings& AegeanSimulation::aegean_sim_settings() { return _aegean_sim_settings; }

        bool AegeanSimulation::has_labels() const
        {
            return (_labels.size() > 0) ? true : false;
        }

    } // namespace simulation
} // namespace simu