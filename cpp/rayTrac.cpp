// rayTrac.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "stdafx.h"


template <typename T>
class Vec3 {

public:
	T x, y, z;
	Vec3() : x(T(0)), y(T(0)), z(T(0)) {};
	Vec3(T a) : x(a), y(a), z(a) {};
	Vec3(T a, T b, T c) : x(a), y(b), z(c) {};

	inline T square_norm() const {
		return x * x + y * y + z * z;
	}

	inline T norm() const {
		return sqrt(square_norm());
	}

	inline Vec3 &normalize() {
		T length = this->norm();
		if (length > 0) {
			T factor = 1.0 / length;
			x *= factor;
			y *= factor;
			z *= factor;
		}
		return *this;
	}

	inline T dot(const Vec3<T>& vec) const {
		return x * vec.x + y * vec.y + z * vec.z;
	}

	inline Vec3<T> &cross(const Vec3<T>& vec) const {
		return Vec3<T>(y * vec.z - z * vec.y, -(x * vec.z - z * vec.x), x * vec.y - y * vec.x);
	}

	Vec3<T> operator + (const Vec3<T> &other) const {
		return Vec3<T>(x + other.x, y + other.y, z + other.z);
	}

	Vec3<T> &operator += (const Vec3<T> &other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	Vec3<T> operator - (const Vec3<T> &other) const {
		return Vec3<T>(x - other.x, y - other.y, z - other.z);
	}

	Vec3<T> operator - () const {
		return Vec3<T>(-x, -y, -z);
	}

	Vec3<T> &operator -= (const Vec3<T> &other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	Vec3<T> operator * (const Vec3<T> &other) const {
		return Vec3<T>(x * other.x, y * other.y, z * other.z);
	}

	Vec3<T> &operator *= (const Vec3<T> &other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		return *this;
	}

	Vec3<T> operator * (const T k) const {
		return Vec3<T>(x * k, y * k, z * k);
	}

	Vec3<T> operator / (const T k) const {
		return Vec3<T>(x / k, y / k, z / k);
	}

	friend std::ostream& operator << (std::ostream & os, const Vec3<T>& vec) {
		os << "( " << vec.x << ", " << vec.y << ", " << vec.z << " )" << std::endl;
		return os;
	}
};

template <typename T>
inline Vec3<T> operator *(const T s, const Vec3<T>& vec) {
	return vec * s;
}


typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3l;



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

const int MAX_DEPTH = 5;


float mix(const float& a, const float& b, const float& mix_rate) {

	return a * (1 - mix_rate) + b * mix_rate;
}

Vec3f ray_trace(const Vec3f& ray_source, const Vec3f &ray_direction, const std::vector<Spheref> &sphere_list, int depth) {

	float bias = 1e-4f;
	float distance = 1e8f;
	Vec3f hit_point;
	bool intersection_bool = false;
	Spheref sphere;
	for (Spheref s : sphere_list) {
		Vec3f tmp_intersection_point;
		bool tmp_intersection_bool;
		tmp_intersection_bool = s.intersection(ray_source, ray_direction, tmp_intersection_point);
		float tmp_distance;
		if (tmp_intersection_bool) {

			intersection_bool = true;
			tmp_distance = (tmp_intersection_point - ray_source).square_norm();
			if (tmp_distance < distance) {
				distance = tmp_distance;
				hit_point = tmp_intersection_point;
				sphere = s;
			}
		}
	}

	if (!intersection_bool) { return Vec3f(2.f, 2.f, 2.f); }

	Vec3f normal_vec = (hit_point - sphere.center).normalize();
	bool is_inside = false;
	if (ray_direction.dot(normal_vec) > 0) {
		// ray_source 在 sphere 内
		normal_vec = -normal_vec;
		is_inside = true;
	}

	Vec3f surface_color(0.f, 0.f, 0.f);
	float fresnel_coeff;
	if ((sphere.transparency > 0 || sphere.reflection > 0) && depth < MAX_DEPTH) {
		float cos_theta = -(normal_vec.dot(ray_direction));


		// 计算反射光线的方向
		Vec3f reflection_direction = ray_direction - normal_vec * (2.0 * ray_direction.dot(normal_vec));
		reflection_direction = reflection_direction.normalize();

		// 稍微偏移, 不让起点在球面上
		Vec3f reflection = ray_trace(hit_point + normal_vec * bias, reflection_direction, sphere_list, depth + 1);


		Vec3f refraction(0.f, 0.f, 0.f);
		float eta=0.f;
		if (sphere.transparency > 0) {
			eta = sphere.refractive_rate;

			if (!is_inside) {
				eta = 1.f / eta;
			}
			float square_cos_alpha = 1 - (1 - cos_theta * cos_theta) * eta * eta;
			if (square_cos_alpha > 0) {
				float cos_alpha;
				Vec3f refraction_direction;
				cos_alpha = sqrt(square_cos_alpha);
				// 计算折射光线方向
				refraction_direction = ray_direction * eta + normal_vec * (eta * cos_theta - cos_alpha);
				refraction_direction = refraction_direction.normalize();
				refraction = ray_trace(hit_point - normal_vec * bias, refraction_direction, sphere_list, depth + 1);
			}
		}
		fresnel_coeff = mix(pow(1.f - cos_theta, 3), 1.f, 0.1f);
		surface_color = (reflection * fresnel_coeff + refraction * (1 - fresnel_coeff) * sphere.transparency) * sphere.surface_color;
	}
	else {
		int num = sphere_list.size();
		float transmission=1.;
		Vec3f light_direction, hit_pt;
		for (int i = 0; i < num; i++) {
			Spheref sphere_i = sphere_list[i];
			if (sphere_i.emission_color.x > 0) {
				transmission = 1;
				light_direction = sphere_i.center - hit_point;
				light_direction = light_direction.normalize();
				for (int j = 0; j++; j < num) {

					if (i == j) continue;
					Spheref sphere_j = sphere_list[j];
					if (sphere_j.intersection(hit_point + normal_vec * bias, light_direction, hit_pt)) {
						// 光线被阻挡
						transmission = 0;
						break;
					}
				}

				surface_color += (sphere.surface_color
					* transmission
					* fmax(0.f, normal_vec.dot(light_direction))
					* sphere_i.emission_color);
			}

		}
	}
	return surface_color + sphere.emission_color;
}


int per_pixel_modify(float x) {
	return int(fmax(fmin(x, 1), 0) * 255.99);
}

Vec3<int> pixel_modify(Vec3f pixel) {

	return Vec3<int>(per_pixel_modify(pixel.x), per_pixel_modify(pixel.y), per_pixel_modify(pixel.z));
}

void render(std::vector<Spheref> sphere_list, int width, int height, float view_angle, Vec3f source_point) {
	float inv_width = 1.0 / width;
	float inv_height = 1.0 / height;
	float tan_theta = tan((view_angle/180.)/2. * M_PI);
	float ratio = (float)width / height;

	float x, y;
	Vec3f r_direction, pixel;
	Vec3<int> rgb_vec;

	std::ofstream pic("res.ppm");
	pic << "P3\n" << width << " " << height << "\n255\n";

	for (int i=0; i < height; i++){
		for (int j=0; j < width; j ++) {
			x = (2 * (j + 0.5) * inv_width - 1) * tan_theta * ratio;
			y = (1 - (2 * (i + 0.5) * inv_height)) * tan_theta;
			r_direction = Vec3f(x, y, -1.0f);
			r_direction = r_direction.normalize();
			pixel = ray_trace(source_point, r_direction, sphere_list, 0);
			rgb_vec = pixel_modify(pixel);
			pic << rgb_vec.x << " " << rgb_vec.y << " " << rgb_vec.z << "\n";
		}
	}
	pic.close();
}


int main() {
	Spheref s0(Vec3f(0.0f, -10004.f, -20.f), 10000.f, 1.1f, Vec3f(0.20f, 0.20f, 0.20f), Vec3f(0.f), 0.f, 0.0f);
	Spheref s1(Vec3f(0.0f, 0.f, -20.f), 4.f, 1.1f, Vec3f(1.0f, 0.32f, 0.36f), Vec3f(0.f), 1.f, 0.5f);
	Spheref s2(Vec3f(5.0f, -1.f, -15.f), 2.f, 1.1f, Vec3f(0.90f, 0.76f, 0.46f), Vec3f(0.f), 1.f, 0.0f);
	Spheref s3(Vec3f(5.0f, 0.f, -25.f), 3.f, 1.1f, Vec3f(0.65f, 0.77f, 0.97f), Vec3f(0.f), 1.f, 0.0f);
	Spheref s4(Vec3f(-5.5f, 0.f, -15.f), 3.f, 1.1f, Vec3f(0.90f, 0.90f, 0.90f), Vec3f(0.f), 1.f, 0.0f);
	Spheref s_array[5] = {s0, s1, s2, s3, s4};
	std::vector<Spheref> sphere_list(s_array, s_array + 5);
	//std::cout << "s3 center: " << s3.center << "s3 radius: " << s3.radius << std::endl;
	//std::cout << sphere_list.size() << std::endl;
	//std::cout << 3.0f * s3.center << std::endl;
	//std::cin.get();

	int width = 640;
	int height = 480;
	float view_angle = 30;
	Vec3f source_point(0.f, 0.f, 0.f);
	DWORD start_time, end_time;
	start_time = GetTickCount64();
	render(sphere_list, width, height, view_angle, source_point);
	end_time = GetTickCount64();
	std::cout << "耗时: " << end_time - start_time << std::endl;
	return 0;
}

