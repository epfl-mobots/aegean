#include "aegean_individual.hpp"
#include "aegean_simulation.hpp"

#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>
#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>

namespace simu {
    namespace simulation {

        using namespace aegean;

        struct Params {
            struct CircularCorridor : public defaults::CircularCorridor {
            };
        };

        AegeanIndividual::AegeanIndividual()
        {
            _is_robot = true;
        }

        AegeanIndividual::~AegeanIndividual() {}

        void AegeanIndividual::stimulate(const std::shared_ptr<Simulation> sim)
        {
            using namespace tools;

            if (_is_robot) {
                auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);

                using distance_func_t
                    = defaults::distance_functions::angular<polygons::CircularCorridor<Params>>;

                int num_individuals = asim->individuals().size();

                Eigen::MatrixXd pos(1, (num_individuals - 1) * 2);
                Eigen::MatrixXd vel(1, (num_individuals - 1) * 2);
                int idx = 0;
                for (const IndividualPtr ind : asim->individuals()) {
                    if (_id == ind->id())
                        continue;
                    pos(idx) = ind->position().x;
                    pos(idx + 1) = ind->position().y;
                    vel(idx) = ind->speed().vx;
                    vel(idx + 1) = ind->speed().vy;
                    idx += 2;
                }
                Eigen::MatrixXd epos(1, pos.cols() + 2);
                epos << pos, _position.x, _position.y;
                features::InterIndividualDistance<distance_func_t> iid;
                iid(epos, asim->aegean_sim_settings().timestep);
                polygons::Point p(_position.x, _position.y);
                Eigen::MatrixXd dist_to_walls(1, 2);
                polygons::CircularCorridor<Params> cc;
                dist_to_walls << cc.distance_to_inner_wall(p), cc.distance_to_outer_wall(p);

                Eigen::MatrixXd nn_input(1, (num_individuals - 1) * 4 /*pos & vel*/ + 3 /*feat set*/ + 2 /*focal pos*/);
                nn_input << pos, vel,
                    iid.get().row(0),
                    dist_to_walls,
                    _position.x, _position.y;

                // auto net = asim->network();
            }
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