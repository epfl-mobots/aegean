/*
This is based on the implementation of TextArchive from the limbo library:
https://github.com/resibots/limbo/blob/master/src/limbo/serialize/text_archive.hpp
*/

#ifndef AEGEAN_TOOLS_ARCHIVE_HPP
#define AEGEAN_TOOLS_ARCHIVE_HPP

#include <Eigen/Core>

#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <iostream>

namespace aegean {
    namespace tools {
        class Archive {
        public:
            Archive() : _fmt(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "", "") {}

            void save(const Eigen::MatrixXd& v, const std::string& filename) const
            {
                std::ofstream ofs(filename);
                ofs << v.format(_fmt) << std::endl;
            }

            template <typename M>
            void load(M& m, const std::string& filename, int skip = 0,
                const char& delim = '\t') const
            {
                auto values = load(filename, skip, delim);
                m.resize(values.size(), values[0].size());
                for (size_t i = 0; i < values.size(); ++i)
                    for (size_t j = 0; j < values[i].size(); ++j)
                        m(i, j) = values[i][j];
            }

            std::vector<std::vector<double>> load(const std::string& filename, int skip = 0,
                const char& delim = ' ') const
            {
                std::ifstream ifs(filename.c_str());
                assert(ifs.good() && "Invalid file path");

                std::string line;
                std::vector<std::vector<double>> v;
                while (std::getline(ifs, line)) {
                    if (skip > 0) {
                        --skip;
                        continue;
                    }

                    std::stringstream line_stream(line);
                    std::string cell;
                    std::vector<double> line;
                    while (std::getline(line_stream, cell, delim)) {
                        (cell == "NAN") ? line.push_back(std::nan(cell.c_str()))
                                        : line.push_back(std::stod(cell));
                    }
                    v.push_back(line);
                }
                assert(!v.empty() && "Empty file");
                return v;
            }

        protected:
            Eigen::IOFormat _fmt;
        };

    } // namespace tools
} // namespace aegean

#endif