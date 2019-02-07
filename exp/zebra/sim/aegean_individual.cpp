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
                using euc_distance_func_t
                    = defaults::distance_functions::euclidean;
                features::InterIndividualDistance<distance_func_t> iid;
                polygons::CircularCorridor<Params> cc;
                features::LinearVelocity lvel;
                features::AngularVelocity avel;
                features::DistanceToAgents<euc_distance_func_t> ldist;
                features::DistanceToAgents<distance_func_t> adist;
                features::AngleDifference adif;
                features::LinearVelocityDifference lvdif;
                features::Bearing brng;
                features::Alignment align;

                int num_individuals = asim->individuals().size();
                float dt = sim->sim_settings().timestep;

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

                auto nets = asim->network();
                uint inputs = nets[0]->layers()[0]->input();
                Eigen::MatrixXd nrow(1, inputs);

                Eigen::MatrixXd pos_block;
                if (asim->iteration() > 0) {
                    pos_block = asim->generated_positions()->block(asim->iteration() - 1, 0, 2, asim->generated_positions()->cols());
                }
                else {
                    pos_block = asim->orig_positions()->block(0, 0, 2, asim->generated_positions()->cols());
                }

                iid(pos_block, dt);
                lvel(pos_block, dt);
                avel(pos_block, dt);
                ldist(pos_block, dt);
                adist(pos_block, dt);
                adif(pos_block, dt);
                lvdif(pos_block, dt);
                brng(pos_block, dt);
                align(pos_block, dt);

                Eigen::MatrixXd iids = iid.get();
                Eigen::MatrixXd lvels = lvel.get();
                Eigen::MatrixXd avels = avel.get() / 360;
                // get only the distance from the focal individual to the neighbours
                Eigen::MatrixXd ldists = ldist.get_vec()[_id];
                Eigen::MatrixXd adists = adist.get_vec()[_id] / 360;
                Eigen::MatrixXd lvdifs = lvdif.get_vec()[_id];
                Eigen::MatrixXd adifs = adif.get_vec()[_id] / 360;

                Eigen::MatrixXd reduced_ldist(1, num_individuals - 1);
                Eigen::MatrixXd reduced_adist(1, num_individuals - 1);
                Eigen::MatrixXd reduced_lvdif(1, num_individuals - 1);
                Eigen::MatrixXd reduced_adif(1, num_individuals - 1);
                for (uint l = 0, idx = 0; l < num_individuals; ++l) {
                    if (l == _id)
                        continue;
                    reduced_ldist(idx) = ldists(0, l);
                    reduced_adist(idx) = adists(0, l);
                    reduced_lvdif(idx) = lvdifs(0, l);
                    reduced_adif(idx) = adifs(0, l);
                    ++idx;
                } // excluding the focal individual

                polygons::Point p(_position.x, _position.y);

                nrow << _speed.vx,
                    _speed.vy,
                    reduced_ldist,
                    reduced_adist,
                    reduced_lvdif,
                    reduced_adif,
                    cc.distance_to_outer_wall(p),
                    cc.angle_to_nearest_wall(p, brng.get()(0, _id)) / 360;

                uint label;
                if (asim->has_labels()) {
                    label = asim->labels()(static_cast<int>(asim->iteration()
                        / asim->aegean_sim_settings().aggregate_window));
                }
                else {

                    Eigen::MatrixXd features(1, 2);
                    uint row_idx;
                    (asim->iteration() == 0) ? row_idx = 0 : row_idx = align.get().rows() - 2;
                    features << iid.get()(0), align.get()(row_idx);
                    auto km = asim->kmeans();
                    label = km->predict(features)(0);
                    (*asim->predictions())(asim->iteration(), _id) = label;
                }

                Eigen::MatrixXd prediction;
                prediction = nets[label]->forward(nrow.transpose()).transpose();

                // static thread_local limbo::tools::rgen_double_t rgen(-0.08 * dt,
                //     0.08 * dt); // adding some random noise to the generated position

                _desired_speed.vx = _speed.vx + prediction(0) * dt;
                _desired_speed.vy = _speed.vy + prediction(1) * dt;
                // std::cout << prediction * dt << std::endl;
                _desired_position.x = _position.x + _desired_speed.vx * dt /*+ rgen.rand()*/;
                _desired_position.y = _position.y + _desired_speed.vy * dt /*+ rgen.rand()*/;
                // std::cout << _position.x << " " << _position.y << " " << _desired_position.x << " "
                //   << " " << _desired_position.y << std::endl;
                p.x() = _desired_position.x;
                p.y() = _desired_position.y;
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