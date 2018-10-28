#ifndef AEGEAN_TOOLS_PRIMITIVES_POINT_HPP
#define AEGEAN_TOOLS_PRIMITIVES_POINT_HPP

namespace aegean {
    namespace tools {
        namespace primitives {

            template <typename T>
            struct Point3D {
                Point3D(T x = 0, T y = 0, T z = 0) : _x(x), _y(y), _z(z) {}

                T x() const { return _x; }
                T& x() { return _x; }

                T y() const { return _y; }
                T& y() { return _y; }

                T z() const { return _z; }
                T& z() { return _z; }

                T _x, _y, _z;
            };

            using Point = Point3D<double>;

        } // namespace primitives
    } // namespace tools
} // namespace aegean

#endif