#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_VELOCITY_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_VELOCITY_HPP

#include <tools/find_data.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;

template <typename RetType>
class Velocity : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    Velocity(const PartialExpData& data)
    {
        this->_type = "velocity";
    }

    void operator()(const PartialExpData& data, std::shared_ptr<RetType> ret, std::vector<size_t> idcs, const size_t boot_iter) override
    {
        (*ret)[1]["velocity"]["means"](0, 0) = 42;
    }
};

#endif