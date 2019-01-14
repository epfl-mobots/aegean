#ifndef SIMU_SIMULATION_AEGEAN_AEGEAN_INDIVIDUAL_HPP
#define SIMU_SIMULATION_AEGEAN_AEGEAN_INDIVIDUAL_HPP

#include <simulation/individual.hpp>

#include <Eigen/Core>

namespace simu {
    namespace simulation {
        using namespace types;
        using IndividualPtr = std::shared_ptr<Individual<double, double>>;

        class AegeanIndividual : public Individual<double, double> {
        public:
            AegeanIndividual(
                // const Eigen::MatrixXd& orig_positions,
                // const Eigen::MatrixXd& orig_velocities
            );
            virtual ~AegeanIndividual();

            virtual void stimulate(const std::shared_ptr<Simulation> sim) override;
            virtual void move(const std::shared_ptr<Simulation> sim) override;
        };

        using AegeanIndividualPtr = std::shared_ptr<AegeanIndividual>;

    } // namespace simulation
} // namespace simu

#endif
