#include "aegean_individual.hpp"

#include "aegean_simulation.hpp"

namespace simu {
    namespace simulation {
        AegeanIndividual::AegeanIndividual(
            // const Eigen::MatrixXd& orig_positions,
            // const Eigen::MatrixXd& orig_velocities
        )
        {
            // _is_robot = true;
            // _position.x = _orig_positions(0, 0);
            // _position.y = _orig_positions(0, 1);
            // _speed.vx = _orig_velocities(0, 0);
            // _speed.vy = _orig_velocities(0, 1);
        }

        AegeanIndividual::~AegeanIndividual() {}

        void AegeanIndividual::stimulate(const std::shared_ptr<Simulation> sim)
        {
            // if (_is_robot) {
            //     auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);
            //     for (const IndividualPtr ind : asim->individuals()) {
            //         if (_id == ind->id())
            //             continue;
            //         // TODO: get nn prediction
            //     }
            // }
        }

        void AegeanIndividual::move(const std::shared_ptr<Simulation> sim)
        {
            auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);
            if (_is_robot) {
                float dt = asim->aegean_sim_settings().timestep;
                // TODO: need to check setup bounds ?
                _speed.vx = (_desired_position.x - _position.x) / dt;
                _speed.vy = (_desired_position.y - _position.y) / dt;
                _position = _desired_position;
            }
            else {
                std::shared_ptr<Eigen::MatrixXd> p = asim->orig_positions();
                _position.x = (*p)(asim->iteration(), _id * 2);
                _position.y = (*p)(asim->iteration(), _id * 2 + 1);
                std::shared_ptr<Eigen::MatrixXd> v = asim->orig_velocities();
                _speed.vx = (*v)(asim->iteration(), _id * 2);
                _speed.vy = (*v)(asim->iteration(), _id * 2 + 1);
            }
        }

    } // namespace simulation
} // namespace simu