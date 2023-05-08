#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_INTERINDIVIDUAL_DISTANCE_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_INTERINDIVIDUAL_DISTANCE_HPP

#include <tools/find_data.hpp>
#include <tools/timers.hpp>
#include <histogram/histogram.hpp>
#include <tools/mathtools.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;
using namespace aegean::histogram;

template <typename RetType>
class InterindividualDistance : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    InterindividualDistance(const PartialExpData& data, float hist_lb, float hist_ub, float hist_bin_size)
        : _hist_lb(hist_lb),
          _hist_ub(hist_ub),
          _hist_bin_size(hist_bin_size),
          _h({hist_lb, hist_ub}, hist_bin_size)
    {
        this->_type = "d";
    }

    void operator()(const PartialExpData& data, std::shared_ptr<RetType> ret, std::vector<size_t> idcs, const size_t boot_iter) override
    {
        // reserve memory in the beginning
        size_t mat_size = 0;
        size_t num_inds = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> segments = std::get<0>(data.at(i));
            for (Eigen::MatrixXd p : segments) {
                mat_size += p.rows();
                num_inds = p.cols() / 2;
            }
        }

        assert(num_inds == 2 && "This stat does not support more individuals");

        // store dists for all experiments in fictitious set
        Eigen::MatrixXd pos(mat_size, num_inds * 2);
        Eigen::VectorXd idist_all_inds(mat_size);

        size_t start_row_idx = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> psegments = std::get<0>(data.at(i));
            for (size_t sidx = 0; sidx < psegments.size(); ++sidx) {
                const Eigen::MatrixXd& p = psegments[sidx];
                pos.block(start_row_idx, 0, p.rows(), num_inds * 2) = p;
                start_row_idx += p.rows();
            }
        } // for different exp idcs

        idist_all_inds.block(0, 0, pos.rows(), 1) = ((pos.col(0) - pos.col(2)).array().square()
            + (pos.col(1) - pos.col(3)).array().square())
                                                        .sqrt();
        { // avg
            Eigen::MatrixXd hist = _h(idist_all_inds);
            (*ret)[this->_type]["avg"]["means"](boot_iter, 0) = idist_all_inds.array().mean();
            (*ret)[this->_type]["avg"]["means2"](boot_iter, 0) = std::pow(idist_all_inds.array().mean(), 2.);
            (*ret)[this->_type]["avg"]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
            (*ret)[this->_type]["avg"]["msamples"](boot_iter) = idist_all_inds.size();
        }
    }

protected:
    float _hist_lb;
    float _hist_ub;
    float _hist_bin_size;

    Histogram _h;
};

#endif