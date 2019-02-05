#ifndef AEGEAN_TOOLS_POLYGONS_POLYGON_BASE_HPP
#define AEGEAN_TOOLS_POLYGONS_POLYGON_BASE_HPP

#include <cassert>
#include <tools/primitives/point.hpp>

namespace aegean {
    namespace tools {
        namespace polygons {

            using namespace primitives;

            class PolygonBase {
            protected:
                PolygonBase() {}

            public:
                virtual double min_distance(const Point& p) const { assert(false); }
                virtual double max_distance(const Point& p) const { assert(false); }
                virtual bool in_polygon(const Point& p) const { assert(false); }
            };
        } // namespace polygons
    } // namespace tools
} // namespace aegean

#endif