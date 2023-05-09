#ifndef AEGEAN_TOOLS_TIMERS_HPP
#define AEGEAN_TOOLS_TIMERS_HPP

#include <chrono>
#include <vector>

namespace aegean {
    namespace tools {
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = std::chrono::time_point<Clock>;

        class Timers {
        public:
            Timers(size_t num_timers = 1)
            {
                _timers.resize(1);
            }

            void timer_start(size_t tidx = 0)
            {
                _timers[tidx] = Clock::now();
            }

            long long timer_stop(size_t tidx = 0) const
            {
                return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - _timers[tidx]).count();
            }

        private:
            std::vector<TimePoint> _timers;
        };
    } // namespace tools
} // namespace aegean

#endif