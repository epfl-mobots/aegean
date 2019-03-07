#include <clustering/clusterdp.hpp>

#include <tools/archive.hpp>

#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace aegean;
using namespace clustering;
using namespace tools;

struct Params {
    struct Clusterdp : public defaults::Clusterdp {
        static constexpr double dc = -1;
        static constexpr double p = .02;
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

    using density_t = defaults::Exp;
    // using density_t = defaults::Chi;

    Clusterdp<Params, densityfun<density_t>> cdp;
    cdp.fit(data, 8);
    std::cout << cdp.centroids() << std::endl;
    cdp.save(archive);

    return 0;
}