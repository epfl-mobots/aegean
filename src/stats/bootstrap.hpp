#ifndef AEGEAN_STATS_BOOTSTRAP_HPP
#define AEGEAN_STATS_BOOTSTRAP_HPP

#include <stats/stat.hpp>
#include <tools/timers.hpp>
#include <random>

#ifdef USE_TBB
#include <tbb/tbb.h>
#include <oneapi/tbb/parallel_for.h>
#endif

namespace aegean {
    namespace stats {

        template <typename T, typename DataType, typename RetType>
        class Bootstrap {
            using stat_t = std::shared_ptr<Stat<T, DataType, RetType>>;

        public:
            Bootstrap(size_t num_exps, size_t M, int num_threads = 1)
                : _num_exps(num_exps),
                  _M(M),
                  _num_threads(num_threads),
                  _gen(_rd()),
                  _rgen(0, num_exps - 1) {}

            void run(const DataType& data, std::shared_ptr<RetType> ret, const std::vector<size_t>& exp_idcs)
            {
                std::cout << "Running bootstrap for " << _num_exps << " experiments and " << _M << " iterations" << std::endl;

                _t.timer_start();

#ifdef USE_TBB
                parallel_bootstrap_loop(data, ret, exp_idcs);
#else
                bootstrap_loop(data, ret, exp_idcs);
#endif

                auto elapsed = _t.timer_stop();
                std::cout << "Done in " << elapsed << " ms" << std::endl;
            }

#ifdef USE_TBB
            void parallel_bootstrap_loop(const DataType& data, std::shared_ptr<RetType> ret, const std::vector<size_t>& exp_idcs)
            {
                if (_num_threads < 0) {
                    _num_threads = oneapi::tbb::info::default_concurrency();
                }
                oneapi::tbb::task_arena arena(_num_threads);
                std::cout << "Starting a TBB arean with " << _num_threads << " threads" << std::endl;
                arena.execute([&] {
                    tbb::parallel_for(
                        tbb::blocked_range<int>(0, _M),
                        [&](tbb::blocked_range<int> r) {
                            for (int i = r.begin(); i < r.end(); ++i) {
                                bootstrap_iteration(data, ret, exp_idcs, i);
                            }
                        });
                });
            }
#endif

            void bootstrap_loop(const DataType& data, std::shared_ptr<RetType> ret, const std::vector<size_t>& exp_idcs)
            {
                for (size_t i = 0; i < _M; ++i) {
                    bootstrap_iteration(data, ret, exp_idcs, i);
                }
            }

            void bootstrap_iteration(const DataType& data, std::shared_ptr<RetType> ret, const std::vector<size_t>& exp_idcs, size_t boot_iter)
            {
                std::vector<size_t> random_idcs;
                for (size_t r = 0; r < _num_exps; ++r) {
                    random_idcs.push_back(exp_idcs[_rgen(_gen)]);
                }

                for (stat_t s : _stats) {
                    s->operator()(data, ret, random_idcs, boot_iter);
                }
            }

            Bootstrap& add_stat(std::shared_ptr<Stat<T, DataType, RetType>> stat)
            {
                _stats.push_back(stat);
                return *this;
            }

            const std::vector<stat_t>& stats() const
            {
                return _stats;
            }

            void clear_stats()
            {
                _stats.clear();
            }

        protected:
            size_t _num_exps;
            size_t _M;
            int _num_threads;
            std::vector<stat_t> _stats;

            tools::Timers _t;
            std::random_device _rd;
            std::mt19937 _gen;
            std::uniform_int_distribution<> _rgen;
        };
    } // namespace stats
} // namespace aegean

#endif