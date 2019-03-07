#include <clustering/kmeans.hpp>
#include <clustering/opt/persistence.hpp>
#include <tools/archive.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace aegean;
using namespace clustering;
using namespace opt;
using namespace tools;

struct Params {
    struct KMeans : public defaults::KMeans {
        static constexpr int max_iter = 100;
    };
};

int main()
{
    Archive archive(false);

    Eigen::MatrixXd data;
    archive.load(data, "./src/examples/datasets/unbalance.txt");

    /* Ground truth centroids 

        209948    349963
        539379    299653
        440134    400135
        440754    298283
        491036    349798
        150007    350104
        538884    400947
        179955    380008
    
    */

    Persistence<KMeans<Params, defaults::KMeansPlusPlus>, 2, 10> per;
    uint optimal_k = per.opt_k(data);
    std::cout << "Optimal k: " << optimal_k << std::endl;

    KMeans<Params, defaults::KMeansPlusPlus> km;
    std::vector<Eigen::MatrixXd> clusters = km.fit(data, optimal_k);
    std::cout << km.centroids() << std::endl;
    km.save(archive);

    return 0;
}