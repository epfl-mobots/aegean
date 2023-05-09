#ifndef AEGEAN_HISTOGRAM_HISTOGRAM_HPP
#define AEGEAN_HISTOGRAM_HISTOGRAM_HPP

#include <tools/archive.hpp>

#include <Eigen/Core>
#include <type_traits>

#include <iostream>
#include <fstream>
#include <utility>

using namespace aegean::tools;

namespace aegean {
    namespace histogram {

        class Histogram {
        public:
            template <typename T>
            Histogram(T a, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr)
                : _num_bins(a), _bin_size(-1), _epsilon(0.0001), _custom_bounds(true), _archive(false) {}

            template <typename T>
            Histogram(T a, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr)
                : _num_bins(-1), _bin_size(a), _epsilon(0.0001), _custom_bounds(true), _archive(false) {}

            Histogram(const std::pair<float, float>& bounds, double bin_size)
                : _num_bins(-1), _bin_size(bin_size), _epsilon(0.0001), _custom_bounds(true), _archive(false)
            {
                double range = bounds.second - bounds.first;
                _lbound = bounds.first;
                _ubound = bounds.second;
                _num_bins = range / _bin_size;
            }

            void set_epsilon(double epsilon)
            {
                _epsilon = epsilon;
            }

            Eigen::MatrixXd operator()(const Eigen::MatrixXd& data)
            {
                if (_custom_bounds) {
                    return _construct_hist(data);
                }
                else {
                    if (_num_bins == -1)
                        return _construct_by_bin_size(data);
                    else
                        return _construct_by_num_bins(data);
                }
            }

            void save(const std::string& filename, const std::string& path = ".") const
            {
                assert((_num_bins > 0) && (_bin_size > 0) && "No data provided in order to construct the histogram");
                std::ofstream ofs(path + "/" + filename + "_info.dat");
                ofs << "Number of bins - Bin size" << std::endl;
                ofs << _num_bins << " " << _bin_size << std::endl;
                _archive.save(_hist, path + "/" + filename + "_bins");
            }

        protected:
            int _num_bins;
            double _bin_size;
            double _epsilon;
            bool _custom_bounds;
            Archive _archive;
            Eigen::MatrixXd _hist;
            double _lbound;
            double _ubound;

            Eigen::MatrixXd _construct_by_bin_size(const Eigen::MatrixXd& data)
            {
                double range = data.maxCoeff() - (data.minCoeff() - _epsilon);
                _num_bins = std::ceil(range / _bin_size);
                _lbound = data.minCoeff();
                _ubound = data.maxCoeff();
                return _construct_hist(data);
            }

            Eigen::MatrixXd _construct_by_num_bins(const Eigen::MatrixXd& data)
            {
                double range = data.maxCoeff() - (data.minCoeff() - _epsilon);
                _bin_size = range / _num_bins;
                _lbound = data.minCoeff();
                _ubound = data.maxCoeff();
                return _construct_hist(data);
            }

            Eigen::MatrixXd _construct_hist(const Eigen::MatrixXd& data)
            {
                Eigen::MatrixXd bins = Eigen::MatrixXd::Zero(1, _num_bins);
                // assume each column is a different dataset that needs to be
                // assigned to the same bins
                for (size_t col = 0; col < data.cols(); ++col) {
                    for (size_t row = 0; row < data.rows(); ++row) {
                        int bin_index = static_cast<int>((data(row, col) - _lbound) / _bin_size);
                        if (bin_index >= 0 && bin_index < bins.size()) {
                            bins(bin_index) += 1.0;
                        }
                    }
                }
                _hist = bins;
                return bins;
            }
        };
    } // namespace histogram
} // namespace aegean

#endif