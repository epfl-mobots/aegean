// #include <histogram/histogram.hpp>
// #include <histogram/hellinger_distance.hpp>
#include <tools/find_data.hpp>
#include <stats/bootstrap.hpp>

#include <Eigen/Core>
#include <iostream>
#include <map>

#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "bootstrap_stats/velocity.hpp"

using namespace aegean;
// using namespace aegean::histogram;
using namespace aegean::tools;
using namespace aegean::stats;

struct mytask {
    mytask(size_t n)
        : _n(n)
    {
    }
    void operator()()
    {
        for (int i = 0; i < 1000000; ++i) {} // Deliberately run slow
        std::cerr << "[" << _n << "]";
    }
    size_t _n;
};

struct executor {
    executor(std::vector<mytask>& t)
        : _tasks(t)
    {
    }
    executor(executor& e, tbb::split)
        : _tasks(e._tasks)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& r) const
    {
        for (size_t i = r.begin(); i != r.end(); ++i)
            _tasks[i]();
    }

    std::vector<mytask>& _tasks;
};

int main(int argc, char** argv)
{
    if (argc < 2) {
        assert("Please provide the root path of the find folder" && false);
    }

    ::FindExps exps = {
        // --
        // {"1_Experiment", {"_processed_positions.dat", false, 0.1}},
        // {"2_Simu", {"_generated_virtu_positions.dat", false, 0.12}},
        {"3_Robot", {"_processed_positions.dat", true, 0.1}}
        // --
    };

    FindData fd(argv[1], exps);
    fd.collect();

    ::JointExps to_join = {
        {"1_Experiment",
            {{2, 1},
                {10, 9},
                {13, 12},
                {14, 12}}},

        {"2_Simu", {}},

        {"3_Robot",
            {{2, 1},
                {3, 1},
                {6, 5},
                {8, 7},
                {10, 9},
                {12, 11},
                {14, 13},
                {16, 15},
                {18, 17},
                {20, 19},
                {22, 21},
                {24, 23},
                {25, 23}}}};

    fd.join_experiments(to_join);

#define TEST_SET "3_Robot"

    // return data structure aliases
    using ret_rec_t = std::unordered_map<std::string, Eigen::MatrixXd>;
    using rec_per_stat_t = std::unordered_map<std::string, ret_rec_t>;
    using ret_t = std::map<size_t, rec_per_stat_t>;

    // num bootstrap iters to run
    const size_t bootstrap_iters = 10000;

    auto data = fd.data();
    {
        // init bootstrap obj
        Bootstrap<Eigen::MatrixXd, PartialExpData, ret_t> exp{data[TEST_SET].size(), bootstrap_iters, 8};

        // initialize stats for bootstrap
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> rvel(new Velocity<ret_t>(data[TEST_SET]));

        // add stats to bootstrap
        exp.add_stat(rvel);

        // get dict keys for bootstrap to iterate over them
        std::vector<size_t> idcs;
        for (const auto& k : data[TEST_SET]) {
            idcs.push_back(k.first);
        }

        // initialize return structure
        auto stats = exp.stats();
        ret_t ret;
        for (const size_t idx : idcs) {
            for (auto s : stats) {
                ret[idx][s->type()] = {
                    {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"N", Eigen::MatrixXd::Zero(bootstrap_iters, 360)}};
            }
        }
        std::shared_ptr ret_ptr = std::make_shared<ret_t>(ret);

        // run bootstrap
        exp.run(data[TEST_SET], ret_ptr, idcs);

        double t = (*ret_ptr)[1]["velocity"]["means"](0, 0);
        std::cout << t << std::endl;
    }
    return 0;
}
