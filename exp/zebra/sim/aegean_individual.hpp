#ifndef SIMU_SIMULATION_AEGEAN_AEGEAN_INDIVIDUAL_HPP
#define SIMU_SIMULATION_AEGEAN_AEGEAN_INDIVIDUAL_HPP

#include <simulation/individual.hpp>

namespace simu {
    namespace simulation {
        using namespace types;

        class AegeanIndividual : public Individual<double, double> {
        public:
            AegeanIndividual() : _is_robot(false)
            {
            }
            virtual ~AegeanIndividual() {}

            virtual void stimulate(const std::shared_ptr<Simulation> sim) override;
            virtual void move(const std::shared_ptr<Simulation>) override;

            bool is_robot() const;
            bool& is_robot();

        protected:
            bool _is_robot;
        };

        using AegeanIndividualPtr = std::shared_ptr<AegeanIndividual>;

    } // namespace simulation
} // namespace simu

#endif
