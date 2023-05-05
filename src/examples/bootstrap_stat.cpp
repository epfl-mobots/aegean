// #include <histogram/histogram.hpp>
// #include <histogram/hellinger_distance.hpp>
// #include <tools/archive.hpp>
#include <tools/find_data.hpp>

#include <Eigen/Core>
#include <iostream>
#include <map>

using namespace aegean;
// using namespace aegean::histogram;
using namespace aegean::tools;

int main(int argc, char** argv)
{
    if (argc < 2) {
        assert("Please provide the root path of the find folder" && false);
    }

    ::FindExps exps = {
        {"1_Experiment", {"_processed_positions.dat", false, 0.1}},
        {"2_Simu", {"_generated_virtu_positions.dat", false, 0.12}},
        {"3_Robot", {"_processed_positions.dat", true, 0.1}}};

    // ::FindExps exps = {
    //     {"3_Robot", {"_processed_positions.dat", true}}};

    FindData fd(argv[1], exps);
    fd.collect();

    ::JointExps to_join = {
        {"1_Experiment",
            {{"exp_2", "exp_1"},
                {"exp_10", "exp_9"},
                {"exp_13", "exp_12"},
                {"exp_14", "exp_12"}}},

        {"2_Simu", {}},

        {"1_Experiment",
            {{"exp_2", "exp_1"},
                {"exp_3", "exp_1"},
                {"exp_6", "exp_5"},
                {"exp_8", "exp_7"},
                {"exp_10", "exp_9"},
                {"exp_12", "exp_11"},
                {"exp_14", "exp_13"},
                {"exp_16", "exp_15"},
                {"exp_18", "exp_17"},
                {"exp_20", "exp_19"},
                {"exp_22", "exp_21"},
                {"exp_24", "exp_23"},
                {"exp_25", "exp_23"}}}};

    fd.join_experiments(to_join);

    return 0;
}
