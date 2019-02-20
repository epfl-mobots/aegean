#include "aegean_individual.hpp"
#include "aegean_simulation.hpp"

#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>
#include <tools/mathtools.hpp>
#include <clustering/kmeans.hpp>

#include <features/inter_individual_distance.hpp>
#include <features/linear_velocity.hpp>
#include <features/angular_velocity.hpp>
#include <features/distance_to_agents.hpp>
#include <features/angle_difference.hpp>
#include <features/linear_velocity_difference.hpp>
#include <features/angular_velocity_difference.hpp>
#include <features/radial_velocity.hpp>

#include <limbo/tools/random_generator.hpp>

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
            _invalid_prediction = false;

            if (_is_robot) {
                auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);

                // the complete feature set
                using circular_corridor_t = polygons::CircularCorridor<Params>;
                using distance_func_t = defaults::distance_functions::angular<circular_corridor_t>;
                // using euc_distance_func_t = defaults::distance_functions::euclidean;
                circular_corridor_t cc;

                int num_individuals = asim->individuals().size();
                float dt = sim->sim_settings().timestep;
                auto nets = asim->network();
                uint inputs = nets[0]->layers()[0]->input();

                // append the focal individual's position at the end
                Eigen::VectorXd reduced_pos((num_individuals - 1) * 2);
                Eigen::VectorXd reduced_vel((num_individuals - 1) * 2);
                int idx = 0;
                for (const IndividualPtr ind : asim->individuals()) {
                    if (_id == ind->id())
                        continue;
                    reduced_pos(idx) = ind->position().x;
                    reduced_pos(idx + 1) = ind->position().y;
                    reduced_vel(idx) = ind->speed().vx;
                    reduced_vel(idx + 1) = ind->speed().vy;
                    idx += 2;
                }

                Eigen::VectorXd nrow(inputs);
                nrow <<
                    // net input
                    reduced_pos
                    // _position.x,
                    // _position.y
                    // net input
                    ;

                uint label;
                if (asim->has_labels()) {
                    label = asim->labels()(static_cast<int>(asim->iteration()
                        / asim->aegean_sim_settings().aggregate_window));
                }
                else {
                    label = 0;
                    // Eigen::MatrixXd features(1, 2);
                    // features << iid.get()(1), align.get()(1);
                    // auto km = asim->kmeans();
                    // label = km->predict(features)(0);
                    // (*asim->predictions())(asim->iteration(), _id) = label;
                }

                Eigen::VectorXd prediction = nets[label]->forward(nrow);

                _desired_position.x = prediction(0);
                _desired_position.y = prediction(1);

                polygons::Point p(_desired_position.x, _desired_position.y);
                bool valid = cc.in_polygon(p);
                if (!valid) {
                    // if the generated positions was outside of the setup
                    // then we maintain the individual's position with a bit of noise
                    // for (uint i = 0; i < 1000; ++i) {
                    //     p.x() = _position.x /*+ rgen.rand()*/;
                    //     p.y() = _position.y /*+ rgen.rand()*/;
                    //     valid = cc.in_polygon(p);
                    //     if (valid)
                    //         break;
                    // }
                    // if (!valid) {
                    _invalid_prediction = true;
                    _desired_position = _position;
                    _desired_speed = _speed;
                    // }
                    // else
                    // {
                    // _desired_position.x = p.x();
                    // _desired_position.y = p.y();
                    // }
                }
            }
        }

        void AegeanIndividual::move(const std::shared_ptr<Simulation> sim)
        {
            auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);
            if (_is_robot) {
                float dt = sim->sim_settings().timestep;
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

        bool AegeanIndividual::invalid_prediction() const
        {
            return _invalid_prediction;
        }

    } // namespace simulation
} // namespace simu