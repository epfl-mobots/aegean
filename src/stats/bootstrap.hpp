#ifndef AEGEAN_STATS_BOOTSTRAP_HPP
#define AEGEAN_STATS_BOOTSTRAP_HPP

#include <stats/stat.hpp>
#include <tools/timers.hpp>
#include <random>

namespace aegean {
    namespace stats {

        template <typename T, typename DataType, typename RetType>
        class Bootstrap {
            using stat_t = std::shared_ptr<Stat<T, DataType, RetType>>;

        public:
            Bootstrap(size_t num_exps, size_t M, size_t num_threads = 1)
                : _num_exps(num_exps),
                  _M(M),
                  _num_threads(num_threads),
                  _gen(_rd()),
                  _rgen(0, num_exps - 1) {}

            void run(const DataType& data, std::shared_ptr<RetType> ret, const std::vector<size_t>& exp_idcs)
            {
                std::cout << "Running bootstrap for " << _num_exps << " experiments and " << _M << " iterations" << std::endl;

                _t.timer_start();

                for (size_t i = 0; i < _M; ++i) {
                    std::vector<size_t> random_idcs;
                    for (size_t r = 0; r < _num_exps; ++r) {
                        random_idcs.push_back(exp_idcs[_rgen(_gen)]);
                    }

                    // TODO: perhaps parallel call stats here?
                    for (stat_t s : _stats) {
                        s->operator()(data, ret, random_idcs, i);
                    }
                }

                auto elapsed = _t.timer_stop();
                std::cout << "Done in " << elapsed << " ms" << std::endl;
            }

            Bootstrap& add_stat(std::shared_ptr<Stat<T, DataType, RetType>> stat)
            {
                _stats.push_back(stat);
                return *this;
            }

            const std::vector<stat_t>& stats() const { return _stats; }

        protected:
            size_t _num_exps;
            size_t _M;
            size_t _num_threads;
            std::vector<stat_t> _stats;

            tools::Timers _t;
            std::random_device _rd;
            std::mt19937 _gen;
            std::uniform_int_distribution<> _rgen;
        };
    } // namespace stats
} // namespace aegean

#endif