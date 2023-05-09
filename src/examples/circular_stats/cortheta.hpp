#ifndef AEGEAN_EXAMPLES_BOOTSTRAP_STATS_CORTHETA_HPP
#define AEGEAN_EXAMPLES_BOOTSTRAP_STATS_CORTHETA_HPP

#include <tools/find_data.hpp>
#include <tools/timers.hpp>
#include <histogram/histogram.hpp>
#include <tools/mathtools.hpp>

using namespace aegean;
using namespace aegean::tools;
using namespace aegean::stats;
using namespace aegean::histogram;

template <typename RetType>
class Cortheta : public Stat<Eigen::MatrixXd, PartialExpData, RetType> {
public:
    Cortheta(const PartialExpData& data, float ntcor, float tcor, float timestep)
        : _ntcor(ntcor),
          _tcor(tcor),
          _timestep(timestep)
    {
        this->_type = "cortheta";
    }

    void operator()(const PartialExpData& data, std::shared_ptr<RetType> ret, std::vector<size_t> idcs, const size_t boot_iter) override
    {
        // reserve memory in the beginning
        size_t num_inds = std::get<0>(data.at(idcs[0]))[0].cols() / 2;
        assert(num_inds == 2 && "This metric is only implemented for 2 individuals");

        // correlation vars
        float dtcor = _ntcor * _timestep;
        size_t ntcorsup = _tcor / dtcor;

        // store dists for all experiments in fictitious set
        Eigen::VectorXd ind0 = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd ind1 = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd avg = Eigen::VectorXd::Zero(ntcorsup);
        Eigen::VectorXd num_data = Eigen::VectorXd::Zero(ntcorsup);

        for (size_t i : idcs) {
            std::vector<Eigen::MatrixXd> psegments = std::get<0>(data.at(i));
            std::vector<Eigen::MatrixXd> vsegments = std::get<1>(data.at(i));
            for (size_t sidx = 0; sidx < psegments.size(); ++sidx) {
                const Eigen::MatrixXd& p = psegments[sidx];
                const Eigen::MatrixXd& v = vsegments[sidx];

                // construct angle of incidence mat
                Eigen::MatrixXd theta(p.rows(), 2);
                for (size_t ind = 0; ind < num_inds; ++ind) {
                    // keep separate individual speeds
                    for (size_t r = 0; r < p.rows(); ++r) {
                        float hdg = std::atan2(v(r, ind * 2 + 1), v(r, ind * 2 + 1));
                        theta(r, ind) = angle_to_pipi(
                            hdg - std::atan2(p(r, ind * 2 + 1), p(r, ind * 2 + 1)));
                    }
                }

                Eigen::VectorXd i0c, i1c, nd;
                std::tie(i0c, i1c, nd) = _compute_correlation(theta, dtcor, ntcorsup);
                ind0 += i0c;
                ind1 += i1c;
                num_data += nd;
            } // for same exp but different segments
        } // for different exp idcs

        avg = ind0 + ind1;
        // avg = (ind0 + ind1).array() / (2 * num_data).array();
        // ind0 = ind0.array() / num_data.array();
        // ind1 = ind1.array() / num_data.array();
        std::vector<Eigen::VectorXd> inds = {std::move(ind0), std::move(ind1)};

        { // avg
            (*ret)[this->_type]["avg"]["means"](boot_iter, 0) = avg.array().mean();
            (*ret)[this->_type]["avg"]["means2"](boot_iter, 0) = std::pow(avg.array().mean(), 2.);
            (*ret)[this->_type]["avg"]["cor"].row(boot_iter) = avg;
            (*ret)[this->_type]["avg"]["num_data"].row(boot_iter) = 2 * num_data;
            (*ret)[this->_type]["avg"]["msamples"](boot_iter) = avg.size();
        }

        for (size_t ind = 0; ind < num_inds; ++ind) {
            const std::string key = "ind" + std::to_string(ind);
            (*ret)[this->_type][key]["means"](boot_iter, 0) = inds[ind].array().mean();
            (*ret)[this->_type][key]["means2"](boot_iter, 0) = std::pow(inds[ind].array().mean(), 2.);
            (*ret)[this->_type][key]["cor"].row(boot_iter) = inds[ind];
            (*ret)[this->_type][key]["num_data"].row(boot_iter) = num_data;
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
                        double cor = std::cos(mat(it, 0) - mat(itp, 0));
                        ind0[itcor] += cor;
                    }

                    { // ind1
                        double cor = std::cos(mat(it, 1) - mat(itp, 1));
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
};

#endif