#ifndef AEGEAN_TOOLS_POLYGONS_CIRCULAR_CORRIDOR_HPP
#define AEGEAN_TOOLS_POLYGONS_CIRCULAR_CORRIDOR_HPP

#include "polygon_base.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <tools/mathtools.hpp>

namespace aegean {
    namespace defaults {
        struct CircularCorridor {
            static constexpr double inner_radius = 0.19;
            static constexpr double outer_radius = 0.29;
            static constexpr double center_x = 0.570587;
            static constexpr double center_y = 0.574004;
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

                double angle_to_nearest_wall(const Point& p, double bearing) const
                {
                    using namespace tools;
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    double tangent_brng = std::fmod(atan2(pt(1), pt(0)) + 2 * M_PI, 2 * M_PI) + M_PI_2;
                    tangent_brng *= 180. / M_PI;
                    tangent_brng = std::fmod(tangent_brng, 360);
                    double diff = tangent_brng - bearing;
                    if (abs(diff) > 180)
                        diff = -1 * sgn(diff) * (360 - abs(diff));
                    return diff;
                }

                double min_distance(const Point& p) const override
                {
                    return std::min(distance_to_inner_wall(p), distance_to_outer_wall(p));
                }

                double max_distance(const Point& p) const override
                {
                    return std::max(distance_to_inner_wall(p), distance_to_outer_wall(p));
                }

                bool in_polygon(const Point& p) const override
                {
                    if (distance_to_outer_wall(p) < 0)
                        return false;
                    else if (distance_to_inner_wall(p) < 0)
                        return false;
                    else
                        return true;
                }

                double distance_to_center(const Point& p) const
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    return pt.norm();
                }

                double distance_to_inner_wall(const Point& p) const
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    return pt.norm() - _inner_r;
                }

                double distance_to_outer_wall(const Point& p) const
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    return _outer_r - pt.norm();
                }

                double angle(const Point& p) const
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    double phi = std::fmod(std::atan2(pt(1), pt(0)) + 2 * M_PI, 2 * M_PI) * 180. / M_PI;
                    return phi;
                }

                bool is_valid(const Point& p) const
                {
                    double corridor_len = _outer_r - _inner_r;
                    if ((distance_to_inner_wall(p) <= corridor_len)
                        && (distance_to_outer_wall(p) <= corridor_len)) {
                        return true;
                    }
                    else
                        return false;
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