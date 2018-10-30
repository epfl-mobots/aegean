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
                virtual double min_distace(const Point& p) { assert(false); }
                virtual double max_distace(const Point& p) { assert(false); }
                virtual bool in_polygon(const Point& p) { assert(false); }
            };
        } // namespace polygons
    } // namespace tools
} // namespace aegean

#endif