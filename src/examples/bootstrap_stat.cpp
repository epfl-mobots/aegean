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
        {"1_Experiment", {"_processed_positions.dat", false}},
        {"2_Simu", {"_processed_positions.dat", false}},
        {"3_Robot", {"_processed_positions.dat", true}}};

    FindData fd(argv[1], exps);
    fd.collect();

    // Eigen::MatrixXd data1 = Eigen::MatrixXd::Random(200, 10).array().abs();
    // Eigen::MatrixXd data2 = Eigen::MatrixXd::Random(200, 10).array().abs();

    // // Histogram hb(.1);
    // Histogram hb(std::make_pair(0., 1.), 0.1);
    // Eigen::MatrixXd bins1 = hb(data1);
    // Eigen::MatrixXd bins2 = hb(data2);

    // HellingerDistance hd;
    // std::cout << "Bins 1: " << bins1 << std::endl;
    // std::cout << "Bins 2: " << bins2 << std::endl;
    // std::cout << "Hellinger distance: " << hd(bins1 / bins1.sum(), bins2 / bins2.sum()) << std::endl;

    return 0;
}
