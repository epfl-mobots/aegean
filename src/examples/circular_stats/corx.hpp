#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_CORX_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_CORX_HPP

#include <tools/find_data.hpp>
#include <tools/timers.hpp>
#include <histogram/histogram.hpp>
#include <tools/mathtools.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;
using namespace aegean::histogram;

template <typename RetType>
class Corx : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    Corx(const PartialExpData& data, float ntcor, float tcor, float timestep, float hist_lb, float hist_ub, float hist_bin_size)
        : _ntcor(ntcor),
          _tcor(tcor),
          _timestep(timestep),
          _hist_lb(hist_lb),
          _hist_ub(hist_ub),
          _hist_bin_size(hist_bin_size),
          _h({hist_lb, hist_ub}, hist_bin_size)
    {
        this->_type = "corx";
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

        assert(num_inds == 2 && "This metric is only implemented for 2 individuals");

        // correlation vars
        float dtcor = _ntcor * _timestep;
        size_t ntcorsup = _tcor / dtcor;

        // store dists for all experiments in fictitious set
        Eigen::MatrixXd pos(mat_size, num_inds * 2);

        Eigen::VectorXd ind0 = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd ind1 = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd avg = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd num_data = Eigen::VectorXd::Zero(ntcorsup);

        size_t start_row_idx = 0;
        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> psegments = std::get<0>(data.at(i));
            for (size_t sidx = 0; sidx < psegments.size(); ++sidx) {
                const Eigen::MatrixXd& p = psegments[sidx];

                Eigen::VectorXd i0c, i1c, nd;
                std::tie(i0c, i1c, nd) = _compute_correlation(p, dtcor, ntcorsup);
                ind0 += i0c;
                ind1 += i1c;
                num_data += nd;
            } // for same exp but different segments
        } // for different exp idcs

        avg = (ind0 + ind1).array() / (2 * num_data).array();
        ind0 = ind0.array() / num_data.array();
        ind1 = ind1.array() / num_data.array();
        std::vector<Eigen::VectorXd> inds = {std::move(ind0), std::move(ind1)};

        { // avg
            Eigen::MatrixXd hist = _h(avg);
            (*ret)[this->_type]["avg"]["means"](boot_iter, 0) = avg.array().mean();
            (*ret)[this->_type]["avg"]["means2"](boot_iter, 0) = std::pow(avg.array().mean(), 2.);
            (*ret)[this->_type]["avg"]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
            (*ret)[this->_type]["avg"]["msamples"](boot_iter) = avg.size();
        }

        for (size_t ind = 0; ind < num_inds; ++ind) {
            const std::string key = "ind" + std::to_string(ind);
            Eigen::MatrixXd hist = _h(inds[ind]);
            (*ret)[this->_type][key]["means"](boot_iter, 0) = inds[ind].array().mean();
            (*ret)[this->_type][key]["means2"](boot_iter, 0) = std::pow(inds[ind].array().mean(), 2.);
            (*ret)[this->_type][key]["N"].row(boot_iter).block(0, 0, 1, hist.cols()) = hist.row(0);
        }
    }

protected:
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> _compute_correlation(const Eigen::MatrixXd& mat, float ntcor, size_t ntcorsup)
    {
        Eigen::VectorXd ind0 = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd ind1 = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd num_data = Eigen::VectorXd::Zero(ntcorsup);

        for (size_t it = 0; it < mat.rows(); ++it) {
            for (size_t itcor = 0; itcor < ntcorsup; ++itcor) {
                size_t itp = it + itcor * ntcor;
                if (itp < mat.rows()) {
                    { // ind0
                        double cor = std::pow(mat(it, 1) - mat(itp, 1), 2.)
                            + std::pow(mat(it, 0) - mat(itp, 0), 2.);
                        ind0[itcor] += cor;
                    }

                    { // ind1
                        double cor = std::pow(mat(it, 3) - mat(itp, 3), 2.)
                            + std::pow(mat(it, 2) - mat(itp, 2), 2.);
                        ind1[itcor] += cor;
                    }

                    ++num_data[itcor];
                }
            }
        }

        return {ind0, ind1, num_data};
    }

    float _ntcor;
    float _tcor;
    float _timestep;

    float _hist_lb;
    float _hist_ub;
    float _hist_bin_size;

    Histogram _h;
};

#endif