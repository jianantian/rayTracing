# include "vec3.h"
# include "stdafx.h"

template <typename T>
class Sphere {
public:
	T radius, refractive_rate, reflection, transparency;
	Vec3<T> center, surface_color, emission_color;
	Sphere() {};
	Sphere(Vec3<T> c, T r, T refr_rate, Vec3<T> s_c, Vec3<T> e_c, T refl_rate, T trans_rate) :
		center(c), radius(r), refractive_rate(refr_rate), surface_color(s_c), emission_color(e_c), reflection(refl_rate), transparency(trans_rate)
	{};

	const bool intersection(const Vec3<T>& source_point, const Vec3<T>& ray_direction, Vec3<T>& intersection_point) {

		Vec3<T> ray = center - source_point;

		T projection = ray.dot(ray_direction);

		if (projection < 0) {
			// 方向不对
			return false;
		}

		else {
			Vec3<T> perpendicular_vec = ray - ray_direction * projection;
			T length;
			length = radius * radius - perpendicular_vec.square_norm();
			if (length < 0) {
				// 未相交
				return false;
			}
			else {
				length = sqrt(length);
				T scale_0, scale_1;
				scale_0 = projection - length;
				scale_1 = projection + length;
				if (scale_0 < 0) {
					intersection_point = ray_direction * scale_1 + source_point;
				}
				else {
					intersection_point = ray_direction * scale_0 + source_point;
				}
				return true;
			}

		}
	}
};

typedef Sphere<float> Spheref;
typedef Sphere<double> Spherel;
