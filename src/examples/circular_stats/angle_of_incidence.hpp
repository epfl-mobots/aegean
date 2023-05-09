#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_ANGLE_OF_INCIDENCE_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_ANGLE_OF_INCIDENCE_HPP

#include <tools/find_data.hpp>
#include <tools/timers.hpp>
#include <histogram/histogram.hpp>
#include <tools/mathtools.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;
using namespace aegean::histogram;

template <typename RetType>
class AngleOfIncidence : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    AngleOfIncidence(const PartialExpData& data, float hist_lb, float hist_ub, float hist_bin_size)
        : _hist_lb(hist_lb),
          _hist_ub(hist_ub),
          _hist_bin_size(hist_bin_size),
          _h({hist_lb, hist_ub}, hist_bin_size)
    {
        this->_type = "thetaw";
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

        // store dists for all experiments in fictitious set
        Eigen::MatrixXd pos(mat_size, num_inds * 2);
        Eigen::MatrixXd vel(mat_size, num_inds * 2);

        Eigen::VectorXd theta_all_inds(mat_size * num_inds);
        std::vector<Eigen::VectorXd> ind_theta(num_inds, Eigen::VectorXd(mat_size, 1));

        size_t start_row_idx = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> psegments = std::get<0>(data.at(i));
            std::vector<Eigen::MatrixXd> vsegments = std::get<1>(data.at(i));
            for (size_t sidx = 0; sidx < psegments.size(); ++sidx) {
                const Eigen::MatrixXd& p = psegments[sidx];
                const Eigen::MatrixXd& v = vsegments[sidx];
                pos.block(start_row_idx, 0, p.rows(), num_inds * 2) = p;
                vel.block(start_row_idx, 0, v.rows(), num_inds * 2) = v;
                start_row_idx += p.rows();
            }
        } // for different exp idcs

        for (size_t ind = 0; ind < num_inds; ++ind) {
            // keep separate individual speeds
            for (size_t r = 0; r < pos.rows(); ++r) {
                float hdg = std::atan2(vel(r, ind * 2 + 1), vel(r, ind * 2 + 1));
                ind_theta[ind](r) = angle_to_pipi(
                    hdg - std::atan2(pos(r, ind * 2 + 1), pos(r, ind * 2 + 1)));
            }

            // concat in one vector too for the average over all individuals
            theta_all_inds.block(pos.rows() * ind, 0, ind_theta[ind].size(), 1) = ind_theta[ind];
        }

        theta_all_inds *= 180 / M_PI;
        for (size_t ind = 0; ind < num_inds; ++ind) {
            ind_theta[ind] *= 180 / M_PI;
        }

        { // avg
            Eigen::MatrixXd hist = _h(theta_all_inds);
            (*ret)[this->_type]["avg"]["means"](boot_iter, 0) = theta_all_inds.array().mean();
            (*ret)[this->_type]["avg"]["means2"](boot_iter, 0) = std::pow(theta_all_inds.array().mean(), 2.);
            (*ret)[this->_type]["avg"]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
            (*ret)[this->_type]["avg"]["msamples"](boot_iter) = theta_all_inds.size();
        }

        for (size_t ind = 0; ind < num_inds; ++ind) {
            const std::string key = "ind" + std::to_string(ind);
            Eigen::MatrixXd hist = _h(ind_theta[ind]);
            (*ret)[this->_type][key]["means"](boot_iter, 0) = ind_theta[ind].array().mean();
            (*ret)[this->_type][key]["means2"](boot_iter, 0) = std::pow(ind_theta[ind].array().mean(), 2.);
            (*ret)[this->_type][key]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
        }
    }

protected:
    float _hist_lb;
    float _hist_ub;
    float _hist_bin_size;

    Histogram _h;
};

#endif