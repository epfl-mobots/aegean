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

#include <tools/timers.hpp>
#include <tools/archive.hpp>

#include "circular_stats/velocity.hpp"
#include "circular_stats/distance_to_wall.hpp"
#include "circular_stats/angle_of_incidence.hpp"
#include "circular_stats/interindividual_distance.hpp"
#include "circular_stats/heading_difference.hpp"
#include "circular_stats/viewing_angle.hpp"
#include "circular_stats/corx.hpp"
#include "circular_stats/corv.hpp"
#include "circular_stats/cortheta.hpp"

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;

class SerializeFindExps : public Archive {
public:
    SerializeFindExps(bool create_dir = true)
        : Archive::Archive(create_dir) {}

    template <typename T>
    void serialize(const std::string& exp, const T data)
    {
        const std::vector<std::string> to_store = {"avg", "ind0", "ind1"};
        const std::vector<std::string> correlations = {"corx", "corv", "cortheta"};

        for (auto [stat, mats] : *data) {
            for (const std::string& type : to_store) {

                if (mats.find(type) == mats.end()) {
                    continue;
                }

                save(mats.at(type).at("means"), exp + "-" + stat + "-" + type + "-means");
                save(mats.at(type).at("means2"), exp + "-" + stat + "-" + type + "-means2");

                if ((type == "avg")) {
                    save(mats.at(type).at("msamples"), exp + "-" + stat + "-" + type + "-samples");
                }

                // if not a correlation stat
                if (std::find(correlations.begin(), correlations.end(), stat) == correlations.end()) {
                    Eigen::MatrixXd N;
                    const Eigen::VectorXd& constNrow = mats.at(type).at("N").row(0);
                    int neg_idx = -1;
                    for (size_t i = 0; i < constNrow.size(); ++i) {
                        if (constNrow(i) < 0) {
                            neg_idx = i;
                            break;
                        }
                    }

                    if (neg_idx > 0) {
                        save(mats.at(type).at("N").block(0, 0, mats.at(type).at("N").rows(), neg_idx), exp + "-" + stat + "-" + type + "-N");
                    }
                    else {
                        save(mats.at("avg").at("N"), exp + "-" + stat + "-" + type + "-N");
                    }
                } // else this is a correlation stat
                else {
                    save(mats.at(type).at("cor"), exp + "-" + stat + "-" + type + "-cor");
                    save(mats.at(type).at("num_data"), exp + "-" + stat + "-" + type + "-num_data");
                }
            } // for all types per stat
        } // for all stats
    }

protected:
};

int main(int argc, char** argv)
{
    if (argc < 2) {
        assert("Please provide the root path of the find folder" && false);
    }

    ::FindExps exps = {
        // --
        {"1_Experiment", {"_processed_positions.dat", false, 0.1, 25}},
        {"2_Simu", {"_generated_virtu_positions.dat", false, 0.12, 25}},
        {"3_Robot", {"_processed_positions.dat", true, 0.1, 25}}
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

    // return data structure aliases
    using ret_rec_t = std::unordered_map<std::string, Eigen::MatrixXd>;
    using ret_t = std::unordered_map<
        std::string,
        std::unordered_map<
            std::string,
            ret_rec_t>>;

    // num bootstrap iters to run
    const size_t bootstrap_iters = (argc > 2) ? std::atoi(argv[2]) : 100;

    const int num_threads = (argc > 3) ? std::atoi(argv[3]) : -1;
    std::cout << "Using " << num_threads << " thread(s)" << std::endl;

    auto data = fd.data();

    Timers t;
    SerializeFindExps ser;
    for (auto [k, sdata] : data) {
        t.timer_start();

        // init bootstrap obj
        Bootstrap<Eigen::MatrixXd, PartialExpData, ret_t> exp{sdata.size(), bootstrap_iters, num_threads};

        // initialize stats for bootstrap
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> rvel(new Velocity<ret_t>(sdata, 0., 35., 0.5));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> dtw(new DistanceToWall<ret_t>(sdata, 25, 0., 25., 0.5));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> thetaw(new AngleOfIncidence<ret_t>(sdata, 0., 180., 1));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> idist(new InterindividualDistance<ret_t>(sdata, 0, 50, 0.5));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> phi(new HeadingDifference<ret_t>(sdata, 0., 180., 1));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> psi(new ViewingAngle<ret_t>(sdata, -180., 180., 1));

        // correlations
        float timestep = std::get<2>(exps.at(k));
        float ntcor = 1;
        float tcor = 30;
        float dtcor = ntcor * timestep;
        size_t ntcorsup = tcor / dtcor;

        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> corx(new Corx<ret_t>(sdata, ntcor, tcor, timestep));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> corv(new Corv<ret_t>(sdata, ntcor, tcor, timestep));
        std::shared_ptr<Stat<Eigen::MatrixXd, PartialExpData, ret_t>> cortheta(new Cortheta<ret_t>(sdata, ntcor, tcor, timestep));

        // add stats to bootstrap
        exp
            //
            .add_stat(rvel)
            .add_stat(dtw)
            .add_stat(thetaw)
            .add_stat(idist)
            .add_stat(phi)
            .add_stat(psi)
            .add_stat(corx)
            .add_stat(corv)
            .add_stat(cortheta)
            //
            ;

        // get dict keys for bootstrap to iterate over them
        std::vector<size_t> idcs;
        for (const auto& k : sdata) {
            idcs.push_back(k.first);
        }

        // initialize return structure
        const std::vector<std::string> single_curve = {"d", "phi"};
        const std::vector<std::string> correlations = {"corx", "corv", "cortheta"};

        auto stats = exp.stats();
        ret_t ret;
        for (auto s : stats) {

            if (std::find(correlations.begin(), correlations.end(), s->type()) == correlations.end()) {
                ret[s->type()]["avg"] = {
                    {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"N", -1 * Eigen::MatrixXd::Ones(bootstrap_iters, 365)},
                    {"msamples", Eigen::MatrixXd::Ones(bootstrap_iters, 1)}};

                if (std::find(single_curve.begin(), single_curve.end(), s->type()) == single_curve.end()) {
                    ret[s->type()]["ind0"] = {
                        {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                        {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                        {"N", -1 * Eigen::MatrixXd::Ones(bootstrap_iters, 365)}};

                    ret[s->type()]["ind1"] = {
                        {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                        {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                        {"N", -1 * Eigen::MatrixXd::Ones(bootstrap_iters, 365)}};
                }
            }
            else {
                ret[s->type()]["avg"] = {
                    {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"cor", Eigen::MatrixXd::Zero(bootstrap_iters, ntcorsup)},
                    {"num_data", Eigen::MatrixXd::Zero(bootstrap_iters, ntcorsup)},
                    {"msamples", Eigen::MatrixXd::Ones(bootstrap_iters, 1)}};

                ret[s->type()]["ind0"] = {
                    {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"cor", Eigen::MatrixXd::Zero(bootstrap_iters, ntcorsup)},
                    {"num_data", Eigen::MatrixXd::Zero(bootstrap_iters, ntcorsup)}};

                ret[s->type()]["ind1"] = {
                    {"means", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"means2", Eigen::MatrixXd::Zero(bootstrap_iters, 1)},
                    {"cor", Eigen::MatrixXd::Zero(bootstrap_iters, ntcorsup)},
                    {"num_data", Eigen::MatrixXd::Zero(bootstrap_iters, ntcorsup)}};
            }
        }
        std::shared_ptr ret_ptr = std::make_shared<ret_t>(ret);

        // run bootstrap
        exp.run(sdata, ret_ptr, idcs);

        // serialize results
        ser.serialize(k, ret_ptr);

        auto elapsed = t.timer_stop();
        std::cout << "[" << k << "] done in " << elapsed << " ms" << std::endl;
    } // for all exp cases

    return 0;
}
