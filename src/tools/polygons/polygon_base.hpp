#ifndef POLYGON_BASE_HPP
#define POLYGON_BASE_HPP

#include <tools/primitives/point.hpp>
#include <cassert>

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