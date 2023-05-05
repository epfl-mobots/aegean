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
    //     {"1_Experiment", {"_processed_positions.dat", false, 0.1}}};

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

    return 0;
}
