#ifndef MATHTOOLS_HPP
#define MATHTOOLS_HPP

#include <Eigen/Core>
#include <vector>
#include <algorithm>

namespace aegean {
    namespace tools {

        template <typename T>
        int sgn(T val)
        {
            return (T(0) < val) - (val < T(0));
        }

        void removeRow(Eigen::MatrixXd& matrix, uint row_idx)
        {
            uint num_rows = matrix.rows() - 1;
            uint num_cols = matrix.cols();
            if (row_idx < num_rows)
                matrix.block(row_idx, 0, num_rows - row_idx, num_cols)
                    = matrix.block(row_idx + 1, 0, num_rows - row_idx, num_cols);
            matrix.conservativeResize(num_rows, num_cols);
        }

        void removeCol(Eigen::MatrixXd& matrix, uint col_idx)
        {
            uint num_rows = matrix.rows();
            uint num_cols = matrix.cols() - 1;
            if (col_idx < num_cols)
                matrix.block(0, col_idx, num_rows, num_cols - col_idx)
                    = matrix.block(0, col_idx + 1, num_rows, num_cols - col_idx);
            matrix.conservativeResize(num_rows, num_cols);
        }

        void removeCols(Eigen::MatrixXd& matrix, std::vector<uint> idcs)
        {
            std::sort(idcs.begin(), idcs.end(), [](uint i, uint j) { return i < j; });
            for (uint i = 0; i < idcs.size(); ++i)
                removeCol(matrix, idcs[i] - i);
        }

        void removeRows(Eigen::MatrixXd& matrix, std::vector<uint> idcs)
        {
            std::sort(idcs.begin(), idcs.end(), [](uint i, uint j) { return i < j; });
            for (uint i = 0; i < idcs.size(); ++i)
                removeRow(matrix, idcs[i] - i);
        }

        template <typename M>
        M rollMatrix(const M& matrix, int offset)
        {
            if (!offset)
                return matrix;
            M rolled(matrix.rows(), matrix.cols());

            if (offset > 0)
                offset = offset % matrix.rows();
            else
                offset = matrix.rows() - (-offset % matrix.rows());
            int rest = matrix.rows() - offset;
            rolled.topRows(offset) = matrix.bottomRows(offset);
            rolled.bottomRows(rest) = matrix.topRows(rest);
            return rolled;
        }

    } // namespace tools
} // namespace aegean

#endif