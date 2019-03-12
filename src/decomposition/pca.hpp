#ifndef AEGEAN_DECOMPOSITION_PCA_HPP
#define AEGEAN_DECOMPOSITION_PCA_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <vector>
#include <cassert>

namespace aegean {
    namespace defaults {
        struct PCA {
            static constexpr bool scale = false;
            static constexpr bool center = true;
        };
    } // namespace defaults

    namespace decomposition {
        template <typename Params>
        class PCA {
        public:
            PCA() {}

            Eigen::MatrixXd operator()(const Eigen::MatrixXd& data)
            {
                _N = data.rows();
                _M = data.cols();

                _centered_data = data;
                if (Params::PCA::center) {
                    _means = data.colwise().mean();
                    _centered_data = data.rowwise() - data.colwise().mean();
                }

                if (Params::PCA::scale) {
                    _stds = Eigen::VectorXd(_M);
                    for (uint i = 0; i < _M; ++i) {
                        _stds(i) = std::sqrt((_centered_data.col(i).array() - _centered_data.col(i).mean()).square().sum() / (_centered_data.col(i).size() - 1));
                        _centered_data.col(i) /= _stds(i);
                    }
                }

                _cov = (_centered_data.adjoint() * _centered_data) / (_centered_data.rows() - 1);
                Eigen::EigenSolver<Eigen::MatrixXd> edecomp(_cov);
                _eigenvalues = edecomp.eigenvalues().real();
                _eigenvectors = edecomp.eigenvectors().real();

                std::vector<std::pair<double, Eigen::VectorXd>> eigen_pairs;
                for (uint i = 0; i < _eigenvectors.cols(); ++i)
                    eigen_pairs.push_back(std::make_pair(_eigenvalues(i), _eigenvectors.col(i)));

                std::sort(eigen_pairs.begin(), eigen_pairs.end(), [](const std::pair<double, Eigen::VectorXd>& lhs, const std::pair<double, Eigen::VectorXd>& rhs) { return lhs.first > rhs.first; });

                double cumsum = .0;
                _cumulative = Eigen::VectorXd(_eigenvalues.rows());
                for (unsigned int i = 0; i < eigen_pairs.size(); i++) {
                    _eigenvalues(i) = eigen_pairs[i].first;
                    cumsum += _eigenvalues(i);
                    _cumulative(i) = cumsum;
                    _eigenvectors.col(i) = eigen_pairs[i].second;
                }

                _variance_explained = Eigen::VectorXd(_eigenvectors.cols());
                for (unsigned int i = 0; i < _eigenvalues.rows(); i++) {
                    if (_eigenvalues(i) > 0) {
                        _variance_explained(i) = _eigenvalues(i) / _eigenvalues.sum();
                    }
                }

                return _eigenvectors;
            }

            const Eigen::MatrixXd& eigenvalues() const { return _eigenvalues; }
            const Eigen::MatrixXd eigenvectors(int num_components = -1) const
            {
                if (num_components < 0)
                    num_components = _eigenvectors.cols();
                return _eigenvectors.block(0, 0, _eigenvectors.rows(), num_components);
            }

            const Eigen::MatrixXd eigenvectors_by_threshold(double perc = 0.95) const
            {
                double cumsum = .0;
                for (uint i = 0; i < _variance_explained.rows(); ++i) {
                    cumsum += _variance_explained(i);
                    if (cumsum >= perc)
                        return _eigenvectors.block(0, 0, _eigenvectors.rows(), i);
                }
                return _eigenvectors;
            }

            const Eigen::MatrixXd& covariance() const { return _cov; }
            const Eigen::VectorXd& variance_explained() const { return _variance_explained; }
            const Eigen::MatrixXd& centered_data() const { return _centered_data; }

            const Eigen::MatrixXd transformed(int num_components = -1) const
            {
                if (num_components < 0)
                    num_components = _eigenvectors.cols();
                return _centered_data * _eigenvectors.block(0, 0, _eigenvectors.rows(), num_components);
            }

            const Eigen::MatrixXd to_original_scale(Eigen::MatrixXd data) const
            {
                if (!Params::PCA::center && !Params::PCA::scale)
                    return data;

                for (uint i = 0; i < data.cols(); ++i) {
                    if (Params::PCA::scale)
                        data.col(i) *= _stds(i);
                    if (Params::PCA::center)
                        data.col(i).array() += _means(i);
                }

                return data;
            }

        protected:
            Eigen::MatrixXd _centered_data;
            Eigen::MatrixXd _eigenvalues;
            Eigen::MatrixXd _eigenvectors;
            Eigen::VectorXd _cumulative;
            Eigen::VectorXd _means;
            Eigen::VectorXd _stds;
            Eigen::MatrixXd _cov;
            Eigen::VectorXd _variance_explained;

            size_t _N;
            size_t _M;
        };

    } // namespace decomposition
} // namespace aegean

#endif