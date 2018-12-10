#include <histogram/histogram.hpp>

#include <Eigen/Core>

#include <iostream>

using namespace aegean;
using namespace aegean::histogram;

int main()
{
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(200, 10);

    Histogram hb(.1);
    Eigen::MatrixXi bins = hb(data);
    std::cout << bins << std::endl;
    hb.save();

    return 0;
}
