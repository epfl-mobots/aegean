#ifndef ARCHIVE_HPP
#define ARCHIVE_HPP

#include <string>
#include <vector>
#include <cassert>
#include <sstream>
#include <fstream>
#include <cmath>

#include <iostream>

namespace aegean {
    namespace tools {
        class Archive {
          public:
            /// load an Eigen matrix (or vector)
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

            /// load a 2D std vector
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
        };

    } // namespace tools
} // namespace aegean

#endif