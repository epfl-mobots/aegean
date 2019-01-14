#include "aegean_individual.hpp"

#include "aegean_simulation.hpp"

namespace simu {
    namespace simulation {
        void AegeanIndividual::stimulate(const std::shared_ptr<Simulation> sim)
        {
            auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);
            for (std::shared_ptr<AegeanIndividual> ind : asim->individuals()) {
                if (_id == ind->id())
                    continue;
            }
        }

        void AegeanIndividual::move(const std::shared_ptr<Simulation>)
        {
        }

        bool AegeanIndividual::is_robot() const { return _is_robot; }
        bool& AegeanIndividual::is_robot() { return _is_robot; }

    } // namespace simulation
} // namespace simu