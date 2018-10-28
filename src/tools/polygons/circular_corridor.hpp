#ifndef AEGEAN_TOOLS_POLYGONS_CIRCULAR_CORRIDOR_HPP
#define AEGEAN_TOOLS_POLYGONS_CIRCULAR_CORRIDOR_HPP

#include "polygon_base.hpp"

namespace aegean {
    namespace defaults {
        struct CircularCorridor {
            static constexpr double inner_radius = 0.19;
            static constexpr double outer_radius = 0.29;
            static constexpr double center_x = 0.58;
            static constexpr double center_y = 0.54;
        };
    } // namespace defaults

    namespace tools {
        namespace polygons {

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

                double min_distace(const Point& p) override
                {
                    return std::min(distance_to_inner_wall(p), distance_to_outer_wall(p));
                }

                double max_distace(const Point& p) override
                {
                    return std::max(distance_to_inner_wall(p), distance_to_outer_wall(p));
                }

                bool in_polygon(const Point& p) override
                {
                    if (distance_to_outer_wall(p) < 0)
                        return false;
                    else if (distance_to_inner_wall(p) < 0)
                        return false;
                    else
                        return true;
                }

                double distance_to_inner_wall(const Point& p)
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    return pt.norm() - _inner_r;
                }

                double distance_to_outer_wall(const Point& p)
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    return _outer_r - pt.norm();
                }

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