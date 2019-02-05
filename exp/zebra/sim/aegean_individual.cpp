#include "aegean_individual.hpp"
#include "aegean_simulation.hpp"

#include <tools/polygons/circular_corridor.hpp>
#include <tools/reconstruction/cspace.hpp>
#include <tools/mathtools.hpp>
#include <features/alignment.hpp>
#include <features/inter_individual_distance.hpp>
#include <clustering/kmeans.hpp>

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
                iid(epos, sim->sim_settings().timestep);
                polygons::Point p(_position.x, _position.y);
                Eigen::MatrixXd dist_to_walls(1, 2);
                polygons::CircularCorridor<Params> cc;
                dist_to_walls << cc.distance_to_inner_wall(p), cc.distance_to_outer_wall(p);

                Eigen::MatrixXd nn_input(1, (num_individuals - 1) * 4 /*pos & vel*/ + 3 /*feat set*/ + 2 /*focal pos*/);
                nn_input << pos, vel,
                    iid.get().row(0),
                    dist_to_walls,
                    _position.x, _position.y;

                auto nets = asim->network();
                uint label;
                if (asim->has_labels()) {
                    label = asim->labels()(static_cast<int>(asim->iteration()
                        / asim->aegean_sim_settings().aggregate_window));
                }
                else {
                    Eigen::MatrixXd pos_block;
                    if (asim->iteration() > 0) {
                        pos_block = asim->generated_positions()->block(asim->iteration() - 1, 0, 2, asim->generated_positions()->cols());
                    }
                    else {
                        pos_block = asim->orig_positions()->block(0, 0, 2, asim->generated_positions()->cols());
                    }

                    features::Alignment align;
                    align(pos_block, sim->sim_settings().timestep);

                    Eigen::MatrixXd features(1, 2);
                    uint row_idx;
                    (asim->iteration() == 0) ? row_idx = 0 : row_idx = align.get().rows() - 2;
                    features << iid.get()(0), align.get()(row_idx);
                    auto km = asim->kmeans();
                    label = km->predict(features)(0);
                    (*asim->predictions())(asim->iteration(), _id) = label;
                }

                Eigen::MatrixXd pos_prediction;
                pos_prediction = nets[label]->forward(nn_input.transpose()).transpose();

                static thread_local limbo::tools::rgen_double_t rgen(-0.08 * asim->sim_settings().timestep,
                    0.08 * asim->sim_settings().timestep); // adding some random noise to the generated position

                _desired_position.x = _position.x + pos_prediction(0) + rgen.rand();
                _desired_position.y = _position.y + pos_prediction(1) + rgen.rand();
                p.x() = _desired_position.x;
                p.y() = _desired_position.y;
                bool valid = cc.in_polygon(p);
                if (!valid) {
                    // if the generated positions was outside of the setup
                    // then we maintain the individual's position with a bit of noise
                    for (uint i = 0; i < 1000; ++i) {
                        p.x() = _position.x + rgen.rand();
                        p.y() = _position.y + rgen.rand();
                        valid = cc.in_polygon(p);
                        if (valid)
                            break;
                    }
                    if (!valid) {
                        _invalid_prediction = true;
                        _desired_position = _position;
                    }
                    else {
                        _desired_position.x = p.x();
                        _desired_position.y = p.y();
                    }
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