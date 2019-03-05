#include <clustering/kmeans.hpp>
#include <clustering/opt/gap_statistic.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace aegean;
using namespace clustering;
using namespace opt;

int main()
{
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(1000, 4);

    GapStatistic<KMeans<defaults::KMeansPlusPlus>, 2, 10> gs;
    uint optimal_k = gs.opt_k(data);

    KMeans<defaults::KMeansPlusPlus> km;
    std::vector<Eigen::MatrixXd> clusters = km.fit(data, optimal_k);
    std::cout << km.centroids() << std::endl;

    return 0;
}