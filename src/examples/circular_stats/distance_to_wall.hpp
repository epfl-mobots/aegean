#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_DISTANCE_TO_WALL_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_DISTANCE_TO_WALL_HPP

#include <tools/find_data.hpp>
#include <tools/timers.hpp>
#include <histogram/histogram.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;
using namespace aegean::histogram;

template <typename RetType>
class DistanceToWall : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    DistanceToWall(const PartialExpData& data, float radius, float hist_lb, float hist_ub, float hist_bin_size)
        : _radius(radius),
          _hist_lb(hist_lb),
          _hist_ub(hist_ub),
          _hist_bin_size(hist_bin_size),
          _h({hist_lb, hist_ub}, hist_bin_size)
    {
        this->_type = "dw";
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
        Eigen::VectorXd dist_all_inds(mat_size * num_inds);
        std::vector<Eigen::VectorXd> ind_dists(num_inds, Eigen::VectorXd(mat_size, 1));

        size_t start_row_idx = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> segments = std::get<0>(data.at(i));
            for (Eigen::MatrixXd p : segments) {
                pos.block(start_row_idx, 0, p.rows(), num_inds * 2) = p;
                start_row_idx += p.rows();
            }
        } // for different exp idcs

        for (size_t ind = 0; ind < num_inds; ++ind) {
            // keep separate individual speeds
            Eigen::VectorXd distance = _radius
                - (pos.col(ind * 2).array().square()
                    + pos.col(ind * 2 + 1).array().square())
                      .array()
                      .sqrt();

            ind_dists[ind] = distance;
            // concat in one vector too for the average over all individuals
            dist_all_inds.block(pos.rows() * ind, 0, distance.size(), 1) = distance;
        }

        { // avg
            Eigen::MatrixXd hist = _h(dist_all_inds);
            (*ret)[this->_type]["avg"]["means"](boot_iter, 0) = dist_all_inds.array().mean();
            (*ret)[this->_type]["avg"]["means2"](boot_iter, 0) = std::pow(dist_all_inds.array().mean(), 2.);
            (*ret)[this->_type]["avg"]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
            (*ret)[this->_type]["avg"]["msamples"](boot_iter) = dist_all_inds.size();
        }

        for (size_t ind = 0; ind < num_inds; ++ind) {
            const std::string key = "ind" + std::to_string(ind);
            Eigen::MatrixXd hist = _h(ind_dists[ind]);
            (*ret)[this->_type][key]["means"](boot_iter, 0) = ind_dists[ind].array().mean();
            (*ret)[this->_type][key]["means2"](boot_iter, 0) = std::pow(ind_dists[ind].array().mean(), 2.);
            (*ret)[this->_type][key]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
        }
    }

protected:
    float _radius;
    float _hist_lb;
    float _hist_ub;
    float _hist_bin_size;

    Histogram _h;
};

#endif