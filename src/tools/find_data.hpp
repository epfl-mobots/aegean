#ifndef AEGEAN_TOOLS_FIND_DATA_HPP
#define AEGEAN_TOOLS_FIND_DATA_HPP

#include <tools/archive.hpp>
#include <tools/mathtools.hpp>

#include <Eigen/Core>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

// for benchmarking
#include <chrono>

namespace aegean {
    namespace tools {
        using FindExps = std::unordered_map<std::string, std::tuple<std::string, bool, float>>;
        using JointExps = std::unordered_map<
            std::string,
            std::unordered_map<std::string, std::string>>;

        using Trajectory = std::pair<Eigen::MatrixXd, int>;
        using TrajectoryWithPath = std::unordered_map<
            std::string,
            Trajectory>;
        using TrajData = std::unordered_map<std::string, TrajectoryWithPath>;
        using VelData = TrajData;

        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = std::chrono::time_point<Clock>;

        class FindData {
        public:
            FindData(const std::string& exp_root, const FindExps& exp_list)
                : _exp_root{exp_root + '/'}, _exps{exp_list}, _arch{false}
            {
            }

            void collect()
            {
                _timer_start();

                for (auto const& [key, val] : _exps) {
                    std::cout << "Collecting data for " << key << ": ";
                    size_t num_files = _retrieve(key);
                    std::cout << "Done [" << num_files << " files]" << std::endl;
                }

                auto elapsed = _timer_stop();
                std::cout << "Data collected in " << elapsed << " ms" << std::endl;
            }

            void join_experiments(const JointExps& to_join)
            {
                _timer_start();

                for (auto const& [key, val] : to_join) {
                    if (_traj.find(key) == _traj.end()) {
                        continue;
                    }
                    else {
                        const TrajectoryWithPath exp_traj = _traj[key];
                        std::vector<std::string> sorted_keys;
                        std::transform(exp_traj.begin(), exp_traj.end(),
                            std::back_inserter(sorted_keys),
                            [](const std::pair<std::string, Trajectory> pair) {
                                return pair.first;
                            });

                        // TODO: there is a nicer version of this but requires newer versions of
                        // TODO: clang for cpp20+ that Apple is not including by default yet ...
                        std::sort(sorted_keys.begin(), sorted_keys.end(),
                            [&](const std::string& lhs, const std::string& rhs) {
                                std::string l_exp_num = _split(lhs, '_')[1];
                                std::string r_exp_num = _split(rhs, '_')[1];

                                if (_is_number(l_exp_num) && _is_number(r_exp_num)) {
                                    int ln = stoi(l_exp_num);
                                    int rn = stoi(r_exp_num);
                                    return ln < rn;
                                }
                                else if (!(_is_number(l_exp_num) && _is_number(r_exp_num))) {
                                    auto ltokens = _split(l_exp_num, '-');
                                    auto rtokens = _split(r_exp_num, '-');
                                    int ln_major = stoi(ltokens[0]);
                                    int ln_minor = stoi(ltokens[1]);
                                    int rn_major = stoi(rtokens[0]);
                                    int rn_minor = stoi(rtokens[1]);

                                    if (ln_major != rn_major) {
                                        return ln_major < rn_major;
                                    }
                                    else {
                                        return ln_minor < rn_minor;
                                    }
                                }
                                else {
                                    // TODO: need to handle this. It might not appear in our data,
                                    // TODO: but it may very well appear in other datasets
                                    assert("Not implemented yet" && false);
                                }
                                return true;
                            });

                        // TODO: update traj data maps
                    }
                }

                auto elapsed = _timer_stop();
                std::cout << "Data joined in " << elapsed << " ms" << std::endl;
            }

        private:
            size_t _retrieve(const std::string& key)
            {
                const std::string path = _exp_root + key;

                boost::filesystem::directory_iterator end_itr;
                for (boost::filesystem::directory_iterator i(path); i != end_itr; ++i) {
                    if (!boost::filesystem::is_regular_file(i->status())) {
                        continue;
                    }

                    std::string exp_file = i->path().filename().string();
                    if (exp_file.ends_with(std::get<0>(_exps[key]))) {
                        std::string full_path = path + "/" + exp_file;
                        Eigen::MatrixXd traj;

                        // load up trajectory matrix
                        _arch.load(traj, full_path);
                        // !! need to compute velocity here, before concat the distros

                        float timestep = std::get<2>(_exps[key]);
                        Eigen::MatrixXd rolled = tools::rollMatrix(traj, 1);
                        Eigen::MatrixXd vel = (traj - rolled) / timestep;
                        vel.row(0) = vel.row(1); // duplicate the first speed to avoid big values (the first velocity can't be computed without t-1 step)

                        int ridx = -1;
                        if (std::get<1>(_exps[key])) {
                            std::string ridx_f = std::regex_replace(
                                exp_file,
                                std::regex(".dat"), "_ridx.dat");
                            std::string full_path = path + "/" + ridx_f;
                            Eigen::VectorXd vidx;
                            _arch.load(vidx, full_path);

                            // load idx file
                            ridx = static_cast<int>(vidx(0));
                        }

                        _traj[key][exp_file] = {traj, ridx};
                        _vels[key][exp_file] = {vel, ridx};
                    }
                }

                return _traj[key].size();
            }

            std::vector<std::string> _split(const std::string& s, char delim)
            {
                std::vector<std::string> tokens;
                std::string token;
                std::istringstream token_stream(s);
                while (std::getline(token_stream, token, delim)) {
                    tokens.push_back(token);
                }
                return tokens;
            }

            bool _is_number(const std::string& s)
            {
                return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
            }

            void _timer_start()
            {
                _t_start = Clock::now();
            }

            long long _timer_stop() const
            {
                return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - _t_start).count();
            }

            TimePoint _t_start;

            std::string _exp_root;
            FindExps _exps;
            Archive _arch;

            TrajData _traj;
            VelData _vels;
        };
    } // namespace tools
} // namespace aegean

#endif