#include <histogram/histogram.hpp>
#include <histogram/hellinger_distance.hpp>
#include <tools/archive.hpp>

#include <Eigen/Core>
#include <iostream>

using namespace aegean;
using namespace aegean::histogram;
using namespace aegean::tools;

int main(int argc, char** argv)
{
    if (argc == 1) {
        Eigen::MatrixXd data1 = Eigen::MatrixXd::Random(200, 10).array().abs();
        Eigen::MatrixXd data2 = Eigen::MatrixXd::Random(200, 10).array().abs();

        // Histogram hb(.1);
        Histogram hb(std::make_pair(0., 1.), 0.1);
        Eigen::MatrixXd bins1 = hb(data1);
        Eigen::MatrixXd bins2 = hb(data2);

        HellingerDistance hd;
        std::cout << "Bins 1: " << bins1 << std::endl;
        std::cout << "Bins 2: " << bins2 << std::endl;
        std::cout << "Hellinger distance: " << hd(bins1 / bins1.sum(), bins2 / bins2.sum()) << std::endl;
    }
    else {
    }

    return 0;
}
