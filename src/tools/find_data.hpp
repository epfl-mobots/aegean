#ifndef AEGEAN_TOOLS_FIND_DATA_HPP
#define AEGEAN_TOOLS_FIND_DATA_HPP

// #include <Eigen/Core>

// #include <cassert>
// #include <cmath>
// #include <fstream>
// #include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <regex>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

// for benchmarking
#include <chrono>

namespace aegean {
    namespace tools {
        using FindExps = std::unordered_map<std::string, std::pair<std::string, bool>>;
        using Ridcs = std::unordered_map<
            std::string,
            std::unordered_map<
                std::string,
                std::pair<std::string, int>>>;
        using Clock = std::chrono::high_resolution_clock;

        class FindData {
        public:
            FindData(const std::string& exp_root, const FindExps& exp_list)
                : _exp_root(exp_root + '/'), _exps(exp_list)
            {
            }

            void collect()
            {
                auto t1 = Clock::now();

                for (auto const& [key, val] : _exps) {
                    std::cout << "Collecting data for " << key << ": ";
                    size_t num_files = _retrieve(key);
                    std::cout << "Done [" << num_files << " files]" << std::endl;
                }

                auto t2 = Clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                std::cout << "Data collected in " << elapsed << " ms" << std::endl;
            }

        private:
            size_t _retrieve(const std::string& key)
            {
                const std::string path = _exp_root + key;
                std::vector<std::string> match;

                boost::filesystem::directory_iterator end_itr;
                for (boost::filesystem::directory_iterator i(path); i != end_itr; ++i) {
                    if (!boost::filesystem::is_regular_file(i->status())) {
                        continue;
                    }

                    std::string exp_file = i->path().filename().string();
                    if (exp_file.ends_with(_exps[key].first)) {
                        match.push_back(exp_file);
                        if (_exps[key].second) {
                            std::string ridx_f = std::regex_replace(
                                exp_file,
                                std::regex(".dat"), "_ridx.dat");
                            _ridcs[key][exp_file] = {ridx_f, -1};
                        }
                    }
                }

                return match.size();
            }

            std::string _exp_root;
            FindExps _exps;
            Ridcs _ridcs;
        };
    } // namespace tools
} // namespace aegean

#endif