#include "aegean_simulation.hpp"

namespace simu {
    namespace simulation {
        AegeanSimulation::AegeanSimulation(NNVec network,
            std::shared_ptr<KMeans<>> kmeans,
            std::shared_ptr<Eigen::MatrixXd> positions,
            std::shared_ptr<Eigen::MatrixXd> velocities,
            std::shared_ptr<Eigen::MatrixXd> predictions,
            std::shared_ptr<Eigen::MatrixXd> generated_positions,
            const std::vector<int>& robot_idcs,
            const Eigen::MatrixXd& labels)
            : Simulation(false),
              _network(network),
              _kmeans(kmeans),
              _positions(positions),
              _velocities(velocities),
              _predictions(predictions),
              _generated_positions(generated_positions),
              _robot_idcs(robot_idcs),
              _labels(labels)
        {
            assert(_positions->rows() > 0);
            assert(_velocities->rows() > 0);
            assert(_positions->rows() == _velocities->rows());
            assert(_positions->cols() == _velocities->cols());

            _init();
            Simulation::_init();
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

            // update the current positions acccording to
            // the most recent prediction
            for (const int idx : idcs) {
                (*_generated_positions)(_iteration, idx * 2) = _individuals[idx]->position().x;
                (*_generated_positions)(_iteration, idx * 2 + 1) = _individuals[idx]->position().y;
            }

            // stimulate the fish to drive their movement decisions
            for (const int idx : idcs) {
                _individuals[idx]->stimulate(std::make_shared<AegeanSimulation>(*this));
            }

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
            _aegean_sim_settings.num_agents = static_cast<int>(_positions->cols() / 2);
            _aegean_sim_settings.num_robot = _robot_idcs.size();
            _aegean_sim_settings.num_fish = _aegean_sim_settings.num_agents - _robot_idcs.size();
            assert(_aegean_sim_settings.num_fish > 0);
            assert(_aegean_sim_settings.num_agents == _aegean_sim_settings.num_robot + _aegean_sim_settings.num_fish);
            if (_robot_idcs.size()) {
                assert(_aegean_sim_settings.num_agents >= *std::max_element(std::begin(_robot_idcs), std::end(_robot_idcs)));
            }
            if (_labels.size()) {
                assert(_labels.rows() == _positions->rows());
            }
            _sim_settings.stats_enabled = _aegean_sim_settings.stats_enabled;
            sim_time() = _positions->rows();

            _individuals.resize(static_cast<size_t>(_aegean_sim_settings.num_agents));
            for (size_t i = 0; i < _individuals.size(); ++i) {
                _individuals[i] = std::make_shared<AegeanIndividual>();
                _individuals[i]->id() = static_cast<int>(i);
                _individuals[i]->position().x = (*_positions)(0, i * 2);
                _individuals[i]->position().y = (*_positions)(0, i * 2 + 1);

                if (_robot_idcs.size() && std::find(_robot_idcs.begin(), _robot_idcs.end(), i) != _robot_idcs.end()) {
                    _individuals[i]->is_robot() = true;
                }
                else {
                    _individuals[i]->is_robot() = false;
                }
            }

            *_generated_positions = Eigen::MatrixXd::Zero(sim_time(), _positions->cols());
            *_predictions = Eigen::MatrixXd::Ones(sim_time(), _aegean_sim_settings.num_agents) * -1;
        }

        const NNVec AegeanSimulation::network() const { return _network; }

        const std::shared_ptr<const Eigen::MatrixXd> AegeanSimulation::orig_positions() const { return _positions; }
        std::shared_ptr<Eigen::MatrixXd> AegeanSimulation::orig_positions() { return _positions; }

        const std::shared_ptr<const Eigen::MatrixXd> AegeanSimulation::orig_velocities() const { return _velocities; }
        std::shared_ptr<Eigen::MatrixXd> AegeanSimulation::orig_velocities() { return _velocities; }

        const std::shared_ptr<const Eigen::MatrixXd> AegeanSimulation::generated_positions() const { return _generated_positions; }
        std::shared_ptr<Eigen::MatrixXd> AegeanSimulation::generated_positions() { return _generated_positions; }

        const std::shared_ptr<const Eigen::MatrixXd> AegeanSimulation::predictions() const { return _predictions; }
        std::shared_ptr<Eigen::MatrixXd> AegeanSimulation::predictions() { return _predictions; }

        std::vector<IndividualPtr> AegeanSimulation::individuals() const { return _individuals; }
        std::vector<IndividualPtr>& AegeanSimulation::individuals() { return _individuals; }

        AegeanSimSettings AegeanSimulation::aegean_sim_settings() const { return _aegean_sim_settings; }
        AegeanSimSettings& AegeanSimulation::aegean_sim_settings() { return _aegean_sim_settings; }

        bool AegeanSimulation::has_labels() const
        {
            return (_labels.size() > 0) ? true : false;
        }

        const Eigen::MatrixXd& AegeanSimulation::labels() const { return _labels; }

        const std::shared_ptr<KMeans<>> AegeanSimulation::kmeans() const { return _kmeans; }

    } // namespace simulation
} // namespace simu