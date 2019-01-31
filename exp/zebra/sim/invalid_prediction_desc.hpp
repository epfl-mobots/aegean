#ifndef SIMU_INVALID_PREDICTION_DESC_HPP
#define SIMU_INVALID_PREDICTION_DESC_HPP

#include "aegean_simulation.hpp"
#include <descriptors/descriptor_base.hpp>

#include <complex>

namespace simu {
    namespace desc {
        using namespace simulation;

        class InvalidPrediction : public DescriptorBase {
        public:
            InvalidPrediction(bool robot_only = false) : _robot_only(robot_only) {}

            virtual void operator()(const std::shared_ptr<Simulation> sim) override
            {
                auto asim = std::dynamic_pointer_cast<AegeanSimulation>(sim);

                if (_desc.size() == 0) {
                    _desc.resize(1, 0.);
                }
                else {
                    int num_individuals = 0;
                    int count_invalid = 0;
                    for (const auto& ind : asim->individuals()) {
                        if (_robot_only && !ind->is_robot())
                            continue;
                        ++num_individuals;
                        auto aind = std::dynamic_pointer_cast<AegeanIndividual>(ind);
                        if (aind->invalid_prediction())
                            ++count_invalid;
                    }
                    _desc[0] += (count_invalid / num_individuals) / static_cast<float>(asim->sim_time());
                }
            }

        protected:
            bool _robot_only;
        };

    } // namespace desc
} // namespace simu

#endif