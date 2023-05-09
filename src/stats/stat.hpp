#ifndef AEGEAN_STATS_DISTRIBUTION_HPP
#define AEGEAN_STATS_DISTRIBUTION_HPP

#include <cassert>

namespace aegean {
    namespace stats {
        template <typename T, typename DataType, typename RetType>
        class Stat {
        public:
            Stat() : _type("base") {}
            virtual ~Stat() {}

            virtual void operator()(const DataType& data, std::shared_ptr<RetType> ret, std::vector<size_t> idcs, const size_t boot_iter)
            {
                assert(false && "The () operator needs to be implemented in derived classes");
            }

            const T& distribution() const { return _dist; }
            const std::string& type() { return _type; }

        protected:
            T _dist;
            std::string _type;
        };
    } // namespace stats
} // namespace aegean

#endif