#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_VELOCITY_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_VELOCITY_HPP

#include <tools/find_data.hpp>
#include <tools/timers.hpp>
#include <histogram/histogram.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;
using namespace aegean::histogram;

template <typename RetType>
class Velocity : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    Velocity(const PartialExpData& data, double hist_lb, double hist_ub, double hist_bin_size)
        : _hist_lb(hist_lb),
          _hist_ub(hist_ub),
          _hist_bin_size(hist_bin_size),
          _h({hist_lb, hist_ub}, hist_bin_size)
    {
        this->_type = "velocity";
    }

    void operator()(const PartialExpData& data, std::shared_ptr<RetType> ret, std::vector<size_t> idcs, const size_t boot_iter) override
    {
        // reserve memory in the beginning
        size_t mat_size = 0;
        size_t num_inds = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> segments = std::get<1>(data.at(i));
            for (Eigen::MatrixXd v : segments) {
                mat_size += v.rows();
                num_inds = v.cols() / 2;
            }
        }

        // store dists for all experiments in fictitious set
        Eigen::MatrixXd vels(mat_size, num_inds * 2);
        Eigen::VectorXd vels_all_inds(mat_size * num_inds);
        std::vector<Eigen::VectorXd> ind_vels(num_inds, Eigen::VectorXd(mat_size, 1));

        size_t start_row_idx = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> segments = std::get<1>(data.at(i));
            for (Eigen::MatrixXd v : segments) {
                vels.block(start_row_idx, 0, v.rows(), num_inds * 2) = v;
                start_row_idx += v.rows();
            }
        } // for different exp idcs

        for (size_t ind = 0; ind < num_inds; ++ind) {
            // keep separate individual speeds
            Eigen::VectorXd speed = (vels.col(ind * 2).array().square()
                + vels.col(ind * 2 + 1).array().square())
                                        .array()
                                        .sqrt();

            ind_vels[ind] = speed;
            // concat in one vector too for the average over all individuals
            vels_all_inds.block(vels.rows() * ind, 0, speed.size(), 1) = speed;
        }

        { // avg
            Eigen::MatrixXd hist = _h(vels_all_inds);
            (*ret)[this->_type]["avg"]["means"](boot_iter, 0) = vels_all_inds.array().mean();
            (*ret)[this->_type]["avg"]["means2"](boot_iter, 0) = std::pow(vels_all_inds.array().mean(), 2.);
            (*ret)[this->_type]["avg"]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
            (*ret)[this->_type]["avg"]["msamples"](boot_iter) = vels_all_inds.size();
        }

        for (size_t ind = 0; ind < num_inds; ++ind) {
            const std::string key = "ind" + std::to_string(ind);
            Eigen::MatrixXd hist = _h(ind_vels[ind]);
            (*ret)[this->_type][key]["means"](boot_iter, 0) = ind_vels[ind].array().mean();
            (*ret)[this->_type][key]["means2"](boot_iter, 0) = std::pow(ind_vels[ind].array().mean(), 2.);
            (*ret)[this->_type][key]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
        }
    }

protected:
    double _hist_lb;
    double _hist_ub;
    double _hist_bin_size;

    Histogram _h;
};

#endif