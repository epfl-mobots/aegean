#ifndef CIRCULAR_CORRIDOR_HPP
#define CIRCULAR_CORRIDOR_HPP

#include "polygon_base.hpp"

namespace aegean {
    namespace tools {
        namespace polygons {

            namespace defaults {
                struct CircularCorridor {
                    static constexpr double inner_radius = 0.19;
                    static constexpr double outer_radius = 0.29;
                    static constexpr double center_x = 0.58;
                    static constexpr double center_y = 0.54;
                };
            } // namespace defaults

            using namespace primitives;

            template <typename Params>
            class CircularCorridor : public PolygonBase {
              public:
                CircularCorridor()
                    : _inner_r(Params::CircularCorridor::inner_radius),
                      _outer_r(Params::CircularCorridor::outer_radius),
                      _center(Params::CircularCorridor::center_x,
                              Params::CircularCorridor::center_y)
                {
                }

                double min_distace(Point p) override {}
                double max_distace(Point p) override {}
                bool in_polygon(Point p) override {}

                double inner_radius() const { return _inner_r; }
                double outer_radius() const { return _outer_r; }
                Point center() const { return _center; }

              protected:
                const double _inner_r;
                const double _outer_r;
                const Point _center;
            };
        } // namespace polygons
    } // namespace tools
} // namespace aegean

#endif