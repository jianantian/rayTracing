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
