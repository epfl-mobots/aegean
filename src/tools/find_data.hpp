#ifndef AEGEAN_TOOLS_FIND_DATA_HPP
#define AEGEAN_TOOLS_FIND_DATA_HPP

#include <tools/archive.hpp>
#include <tools/mathtools.hpp>
#include <tools/timers.hpp>

#include <Eigen/Core>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace aegean {
    namespace tools {
        using FindExps = std::unordered_map<std::string, std::tuple<std::string, bool, float, float>>;
        using JointExps = std::unordered_map<
            std::string,
            std::unordered_map<int, int>>;

        using Trajectory = std::pair<Eigen::MatrixXd, int>;
        using TrajectoryWithPath = std::unordered_map<
            std::string,
            Trajectory>;
        using TrajData = std::unordered_map<std::string, TrajectoryWithPath>;
        using VelData = TrajData;

        using PartialExpData = std::unordered_map<size_t,
            std::tuple<std::vector<Eigen::MatrixXd>,
                std::vector<Eigen::MatrixXd>,
                int>>;
        using ExpData = std::unordered_map<std::string, PartialExpData>;

        class FindData {
        public:
            FindData(const std::string& exp_root, const FindExps& exp_list)
                : _exp_root{exp_root + '/'}, _exps{exp_list}, _arch{false}, _unified_data(false)
            {
            }

            void collect()
            {
                _traj.clear();
                _vels.clear();
                _uni_data.clear();

                _t.timer_start();

                for (auto const& [key, val] : _exps) {
                    std::cout << "Collecting data for " << key << ": ";
                    size_t num_files = _retrieve(key);
                    std::cout << "Done [" << num_files << " files]" << std::endl;
                }

                auto elapsed = _t.timer_stop();
                std::cout << "Data collected in " << elapsed << " ms" << std::endl;
            }

            void join_experiments(const JointExps& to_join)
            {
                _t.timer_start();

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

                        for (const std::string& skey : sorted_keys) {
                            std::string str_exp_num = _split(skey, '_')[1];
                            std::vector<std::string> check_minor = _split(str_exp_num, '-');
                            if (check_minor.size()) {
                                str_exp_num = check_minor[0];
                            }
                            int exp_num = stoi(str_exp_num);

                            auto viter = val.find(exp_num);
                            if (viter == val.end()) {
                                std::get<0>(_uni_data[key][exp_num]).push_back(std::move(_traj[key][skey].first));
                                std::get<1>(_uni_data[key][exp_num]).push_back(std::move(_vels[key][skey].first));
                                std::get<2>(_uni_data[key][exp_num]) = _traj[key][skey].second;
                            }
                            else {
                                int ridx1 = _traj[key][skey].second;
                                int ridx0 = std::get<2>(_uni_data[key][viter->second]);

                                // there is a chance that the split experiments will have different robot indices (this is because
                                // they are indeed analyzed by a different instance of the tracking software). Here, we swap the
                                // columns to make sure that both experiments that are about to be joined have the same robot index
                                if (ridx0 != ridx1) {
                                    _traj[key][skey].first.col(ridx0 * 2).swap(_traj[key][skey].first.col(ridx1 * 2));
                                    _traj[key][skey].first.col(ridx0 * 2 + 1).swap(_traj[key][skey].first.col(ridx1 * 2 + 1));
                                    _vels[key][skey].first.col(ridx0 * 2).swap(_vels[key][skey].first.col(ridx1 * 2));
                                    _vels[key][skey].first.col(ridx0 * 2 + 1).swap(_vels[key][skey].first.col(ridx1 * 2 + 1));
                                    _traj[key][skey].second = ridx0;
                                    _vels[key][skey].second = ridx0;
                                }

                                std::get<0>(_uni_data[key][viter->second]).push_back(std::move(_traj[key][skey].first));
                                std::get<1>(_uni_data[key][viter->second]).push_back(std::move(_vels[key][skey].first));
                            }
                        }

                        _unified_data = true;
                    }
                }

                auto elapsed = _t.timer_stop();
                std::cout << "Data joined in " << elapsed << " ms" << std::endl;
            }

            const TrajData& trajectories() const { return _traj; }
            TrajData& trajectories() { return _traj; }

            const VelData& velocities() const { return _vels; }
            VelData& velocities() { return _vels; }

            const ExpData& data() const { return _uni_data; }
            ExpData& data() { return _uni_data; }

            bool is_data_unified() const { return _unified_data; }

        protected:
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

                        // - compute velocities
                        float timestep = std::get<2>(_exps[key]);
                        float radius = std::get<3>(_exps[key]);

                        traj *= radius;
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

            Timers _t;

            std::string _exp_root;
            FindExps _exps;
            Archive _arch;
            bool _unified_data;

            TrajData _traj;
            VelData _vels;
            ExpData _uni_data;
        };
    } // namespace tools
} // namespace aegean

#endif