#ifndef AEGEAN_TOOLS_POLYGONS_CIRCULAR_OPEN_HPP
#define AEGEAN_TOOLS_POLYGONS_CIRCULAR_OPEN_HPP

#include "polygon_base.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <tools/mathtools.hpp>

namespace aegean {
    namespace defaults {
        struct CircularOpen {
            static constexpr double radius = 0.29;
            static constexpr double center_x = 0.570587;
            static constexpr double center_y = 0.574004;
        };
    } // namespace defaults

    namespace tools {
        namespace polygons {

            using namespace primitives;

            template <typename Params>
            class CircularOpen : public PolygonBase {
            public:
                CircularOpen()
                    : _r(Params::CircularOpen::radius),
                      _center(Params::CircularOpen::center_x,
                          Params::CircularOpen::center_y)
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
                    return distance_to_wall(p);
                }

                bool in_polygon(const Point& p) const override
                {
                    if (distance_to_wall(p) < 0)
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

                double distance_to_wall(const Point& p) const
                {
                    return _r - distance_to_center(p);
                }

                double angle(const Point& p) const
                {
                    Eigen::Vector2d pt(2);
                    pt(0) = p.x() - _center.x();
                    pt(1) = p.y() - _center.y();
                    double phi = std::fmod(std::atan2(pt(1), pt(0)) + 2 * M_PI, 2 * M_PI) * 180. / M_PI;
                    return phi;
                }

                double radius() const { return _r; }
                Point center() const { return _center; }

            protected:
                const double _r;
                const Point _center;
            };
        } // namespace polygons
    } // namespace tools
} // namespace aegean

#endif