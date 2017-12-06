#pragma once

#ifndef _USE_MATH_DEFINES
#define M_PI				3.14159265358979323846
#define M_PI_2				1.57079632679489661923
#define M_PI_4				0.785398163397448309616
#define M_1_PI				0.318309886183790671538
#define M_2_PI				0.636619772367581343076
#define M_2_SQRTPI			1.12837916709551257390
#endif

#define M_RAD_TO_DEG			57.295779513
#define M_DEG_TO_RAD			0.01745329252

#include <cmath>
#include <string>
#include <cstdarg>
#include <array>
#include <float.h>
#include "traits.h"

#define F32MAX FLT_MAX

namespace vian{

	typedef unsigned char u8;
	typedef unsigned short u16;
	typedef unsigned int u32;
	typedef unsigned long long u64;
	
	typedef char s8;
	typedef short s16;
	typedef int s32;
	typedef long long s64;
	
	typedef float f32;
	typedef double f64;

	template<typename T, typename s32 size>
	class Vector {
	public:
		T data[size];

		Vector() : data {} {}

		template<typename U>
		operator Vector<U, size>() {
			Vector<U, size> result;
			for (s32 i = 0; i < size; i++) {
				result.data[i] = static_cast<U>(this->data[i]);
			}
			return result;
		}

		T& operator[](s32 index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 4> {
	public:
		union {
			//Adding more anonymous struct magic...
			T data[4];
			struct { T x, y, z, w; };
			struct { T r, g, b, a; };
			struct { Vector<T, 2> xy; T z, w; };
			struct { T x; Vector<T, 2> yz; T w; };
			struct { T x, y; Vector<T, 2> zw; };
			struct { Vector<T, 2> xy, zw; };
			struct { Vector<T, 3> xyz; T w; };
			struct { T x; Vector<T, 3> yzw; };
			Vector<T, 3> rgb;
		};
		Vector() : data{} { }
		Vector(T x, T y, T z, T w) : data{ x, y ,z ,w } { }
		Vector(const Vector<T, 3>& xyz, T w) : xyz{ xyz }, w{ w } { }
		Vector(T x, const Vector<T, 3>& yzw) : x{ x }, yzw{ yzw } { }
		Vector(const Vector<T, 2>& xy, T z, T w) : xy{ xy }, z{ z }, w{ w } { } 
		Vector(T x, const Vector<T, 2>& yz, T w) : x{ x }, yz{ yz }, w{ w } { }
		Vector(T x, T y, const Vector<T, 2>& zw) : x{ x }, y{ y }, zw{ zw } { }
		Vector(const Vector<T, 2>& xy, const Vector<T, 2>& zw) { }

		template<typename U>
		operator Vector<U, 4>() {
			return Vector<U, 4>(
				static_cast<U>(this->x),
				static_cast<U>(this->y),
				static_cast<U>(this->z),
				static_cast<U>(this->w));
		}

		T& operator[](s32 index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 3> {
	public:
		union {
			T data[3];
			struct { T x, y, z; };
			struct { T r, g, b; };
			struct { T x; Vector<T, 2> yz; };
			struct { Vector<T, 2> xy; T z; };
		};
		Vector() : data{} { }
		Vector(T x, T y, T z) : data{ x, y, z } { }
		Vector(const Vector<T, 2>& xy, T z) : xy{ xy }, z{ z } { }
		Vector(T x, const Vector<T, 2>& yz) : x{ x }, yz{ yz } { }
		template<typename U>
		operator Vector<U, 3>() {
			return Vector<U, 3>(
				static_cast<U>(this->x),
				static_cast<U>(this->y),
				static_cast<U>(this->z));
		}
		T& operator[](s32 index) {
			return data[index];
		}
	};
	
	
	template<typename T>
	class Vector<T, 2> {
	public:
		union {
			T data[2];
			struct { T x, y; };
		};

		Vector() : data{} { }
		Vector(T x, T y) : x{ x }, y{ y } { }

		template<typename U>
		operator Vector<U, 2>() {
			return Vector<U, 2>(
				static_cast<U>(this->x),
				static_cast<U>(this->y));
		}

		T& operator[](s32 index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 1> {
	public:
		union {
			T data[1];
			struct { T x; };
		};

		Vector() : data{} { }
		Vector(T x ) : x{ x } { }
		template<typename U>
		operator Vector<U, 1>() {
			return Vector<U, 1>(static_cast<U>(this->x));
		}

		T& operator[](s32 index) {
			return data[index];
		}
	};

	template<typename s32 size>
	using vecn = Vector<f32, size>;
	using vec4 = Vector<f32, 4>;
	using vec3 = Vector<f32, 3>;
	using vec2 = Vector<f32, 2>;

	template<typename s32 size>
	using vecni = Vector<s32, size>;
	using vec4i = Vector<s32, 4>;
	using vec3i = Vector<s32, 3>;
	using vec2i = Vector<s32, 2>;

	template<typename s32 size>
	using vecnd = Vector<f64, size>;
	using vec4d = Vector<f64, 4>;
	using vec3d = Vector<f64, 3>;
	using vec2d = Vector<f64, 2>;
	
	// Vector + Vector
	template<typename T, typename U, typename s32 size>
	Vector<T, size> operator +(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result;
		for (s32 i = 0; i < size; i++)
			result.data[i] = lhs.data[i] + static_cast<T>(rhs.data[i]);
		return result;
	}

	//Negated Vector
	template<typename T, typename s32 size>
	Vector<T, size> operator -(const Vector<T, size>& rhs) {
		Vector<T, size> result;
		for (s32 i = 0; i < size; i++)
			result.data[i] = -rhs.data[i];
		return result;
	}

	//Vector - Vector
	template<typename T, typename U, typename s32 size>
	Vector<T, size> operator -(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result;
		result = lhs + (-rhs);
		return result;
	}

	//Vector * Scalar
	template<typename T, typename U, typename s32 size>
	Vector<T, size> operator *(const Vector<T, size>& lhs, const U& rhs) {
		static_assert(is_scalar<U>::value, "Vector * scalar multiplication spects a scalar as its right argument"); // should I do static asserts more?
		Vector<T, size> result;
		T s = static_cast<T>(rhs);
		for (s32 i = 0; i < size; i++)
			result.data[i] = lhs.data[i] * s;
		return result;
	}
	
	//Vector * Scalar
	template<typename T, typename U, typename s32 size>
	Vector<T, size> operator /(const Vector<T, size>& lhs, const U& rhs) {
		static_assert(is_scalar<U>::value, "Vector * scalar multiplication spects a scalar as its right argument"); // should I do static asserts more?
		Vector<T, size> result;
		T s = static_cast<T>(rhs);
		for (s32 i = 0; i < size; i++)
			result.data[i] = lhs.data[i] / s;
		return result;
	}

	//Dot product between two vectors
	template<typename T, typename U, typename s32 size>
	T Dot(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		T result = 0;
		for (s32 i = 0; i < size; i++)
			result += lhs.data[i] * static_cast<T>(rhs.data[i]);
		return result;
	}

	template<typename T, typename U, typename s32 size>
	Vector<T, size> Hadamard(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result {};
		for (s32 i = 0; i < size; i++)
			result.data[i] = lhs.data[i] * static_cast<T>(rhs.data[i]);
		return result;
	}

	//Cross product between two vectors
	template<typename T, typename U>
	Vector<T,3> Cross(const Vector<T, 3>& lhs, const Vector<U, 3>& rhs) {
		Vector<T, 3> result;

		result.data[0] = (lhs.data[1] * static_cast<T>(rhs.data[2])) - (lhs.data[2] * static_cast<T>(rhs.data[1]));
		result.data[1] = (lhs.data[2] * static_cast<T>(rhs.data[0])) - (lhs.data[0] * static_cast<T>(rhs.data[2]));
		result.data[2] = (lhs.data[0] * static_cast<T>(rhs.data[1])) - (lhs.data[1] * static_cast<T>(rhs.data[0]));

		return result;
	}



	
	// Magnitude of a N size Vector
	template<typename T, typename s32 size>
	f64 Magnitude(const Vector<T, size>& lhs) {
		f64 result = 0;
		for (s32 i = 0; i < size; i++) {
			result += static_cast<f64>(lhs.data[i] * lhs.data[i]);
		}
		result = sqrt(result);
		return result;
	}

	// Magnitude squared of a N size Vector
	template<typename T, typename s32 size>
	f64 MagnitudeSqr(const Vector<T, size>& lhs) {
		f64 result = 0;
		for (s32 i = 0; i < size; i++) {
			result += static_cast<f64>(lhs.data[i] * lhs.data[i]);
		}
		return result;
	}

	// Normalize a Vector of N size
	template<typename T, typename s32 size>
	vecnd<size> Normalize(const Vector<T, size>& lhs) {
		Vector<f64, size> result;
		f64 mag = vian::Magnitude(lhs);
		for (s32 i = 0; i < size; i++) {
			result.data[i] = static_cast<f64>(lhs.data[i]) / mag;
		}
		return result;
	}

	// Get a vector that pos32s from lhs to rhs
	template<typename T, typename U, typename s32 size>
	Vector<T, size> Direction(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result;
		result = rhs - lhs;
		return result;
	}

	// Get the scalar distance between two vectors of the same size
	template<typename T, typename U, typename s32 size>
	f64 Distance(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		f64 result;
		result = vian::Magnitude(rhs - lhs);
		return result;
	}

	// Get the squared scalar distance between two vectors of the same size
	template<typename T, typename U, typename s32 size>
	f64 DistanceSqr(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		f64 result;
		result = vian::MagnitudeSqr(rhs - lhs);
		return result;
	}

	template<typename T, typename U, typename s32 size>
	Vector<T, size> Lerp(const Vector<T, size>& lhs, f32 t, const Vector<U, size>& rhs){
		Vector<T, size> result;
		result =  (lhs * t) + (rhs * (1.f - t));
		return result;
	}

	// Get the angle between two Vectors of the same size in radians
	template<typename T, typename U, typename s32 size>
	f64 Angle(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		f64 result;
		f64 dot = vian::Dot(lhs, rhs);
		result = dot / (vian::Magnitude(rhs) * vian::Magnitude(lhs));
		result = acos(result);
		return result;
	}

	template<typename T, typename U, typename s32 size>
	f64 AngleDeg(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		f64 result;
		result = vian::Angle(lhs, rhs) * M_RAD_TO_DEG;
		return result;
	}

	template<typename T, s32 rows, s32 cols>
	class Matrix{
	public:
		T data[rows][cols] = {};

		
		Matrix() {
			if (rows == cols) {
				LoadIdentity();
			}
		}

		Matrix(const std::array<std::array<T, cols>, rows>& args) {
				for (s32 i = 0; i < rows; i++) {
					for (s32 j = 0; j < cols; j++) {
						this->data[i][j] = args[i][j];
					}
				}
		}

		template<typename U>
		operator Matrix<U, rows, cols>() {
			Matrix<U, rows, cols> tmp;
			for (s32 i = 0; i < rows; i++) {
				for (s32 j = 0; j < cols; j++) {
					tmp.data[i][j] = static_cast<U>(this->data[i][j]);
				}
			}
			return tmp;
		}

		void LoadIdentity() {
			for (s32 i = 0; i < rows; i++) {
				for (s32 j = 0; j < cols; j++) {
					if (i != j)
						data[i][j] = 0;
					else
						data[i][j] = 1;
				}
			}
		}

		/* NOTE: Should I put these in here?. Maybe a static class for this functions? */

		static Matrix<T, rows, cols> Identity();

		//Returns a matrix translated by a vector
		static Matrix<T, 4, 4> Translation(const Vector<T, 3>& v);
		static Matrix<T, 4, 4> Translation(const T& x, const T& y, const T& z);

		/*
		Returns a matrix rotated by a the angles (radians) given in the x, y and z axis.
		Order of rotation is Y -> Z -> X;
		*/
		static Matrix<T, 4, 4> Rotation(const Vector<T, 3>& v);
		static Matrix<T, 4, 4> Rotation(const T& x, const T& y, const T& z);

		//Returns a matrix scaled by a vector
		static Matrix<T, 4, 4> Scale(const Vector<T, 3>& v);
		static Matrix<T, 4, 4> Scale(const T& x, const T& y, const T& z);
	};

	template<s32 rows, s32 cols>
	using mat = Matrix<f32, rows, cols>;
	template<s32 rows_cols>
	using matn = Matrix<f32, rows_cols, rows_cols>;
	using mat4 = Matrix<f32, 4, 4>;
	using mat3 = Matrix<f32, 3, 3>;
	using mat2 = Matrix<f32, 2, 2>;

	template<s32 rows, s32 cols>
	using matd = Matrix<f64, rows, cols>;
	template<s32 rows_cols>
	using matnd = Matrix<f64, rows_cols, rows_cols>;
	using mat4d = Matrix<f64, 4, 4>;
	using mat3d = Matrix<f64, 3, 3>;
	using mat2d = Matrix<f64, 2, 2>;



	template<typename T, s32 rows, s32 cols>
	Matrix<T, rows, cols> Matrix<T, rows, cols>::Identity() {
		Matrix<T, rows, cols> result;
		if (rows == cols) {
			for (s32 i = 0; i < rows; i++) {
				for (s32 j = 0; j < cols; j++) {
					if (i != j)
						result.data[i][j] = 0;
					else
						result.data[i][j] = 1;
				}
			}
		}
		return result;
	}

	//Returns a matrix translated by a vector
	template<typename T, s32 rows, s32 cols>
	Matrix<T, 4, 4> Matrix<T, rows, cols>::Translation(const Vector<T, 3>& v) {
		return Matrix<T, rows, cols>::Translation(v.x, v.y, v.z);
	}
	template<typename T, s32 rows, s32 cols>
	Matrix<T, 4, 4> Matrix<T, rows, cols>::Translation(const T& x, const T& y, const T& z) {
		Matrix<T, 4, 4> result;
		result.data[0][3] = x;
		result.data[1][3] = y;
		result.data[2][3] = z;
		result.data[3][3] = 1;
		return result;
	}

	/*
	Returns a matrix rotated by the angles (radians) given in the x, y and z axis.
	Order of rotation is Y -> Z -> X;
	*/
	template<typename T, s32 rows, s32 cols>
	Matrix<T, 4, 4> Matrix<T, rows, cols>::Rotation(const Vector<T, 3>& v) {
		return Matrix<T, rows, cols>::Rotation(v.x, v.y, v.z);
	}
	template<typename T, s32 rows, s32 cols>
	Matrix<T, 4, 4> Matrix<T, rows, cols>::Rotation(const T& x, const T& y, const T& z) {
		Matrix<T, 4, 4> result;
		Matrix<T, 4, 4> Rx;
		Matrix<T, 4, 4> Ry;
		Matrix<T, 4, 4> Rz;

		Rx.data[1][1] = cos(x);
		Rx.data[1][2] = -sin(x);
		Rx.data[2][1] = sin(x);
		Rx.data[2][2] = cos(x);

		Ry.data[0][0] = cos(y);
		Ry.data[0][2] = sin(y);
		Ry.data[2][0] = -sin(y);
		Ry.data[2][2] = cos(y);

		Rz.data[0][0] = cos(z);
		Rz.data[0][1] = -sin(z);
		Rz.data[1][0] = sin(z);
		Rz.data[1][1] = cos(z);

		result = Rx * Rz * Ry;

		return result;
	}



	//Returns a matrix scaled by a vector
	template<typename T, s32 rows, s32 cols>
	Matrix<T, 4, 4> Matrix<T, rows, cols>::Scale(const Vector<T, 3>& v) {
		return Matrix<T, rows, cols>::Scale(v.x, v.y, v.z);
	}
	template<typename T, s32 rows, s32 cols>
	Matrix<T, 4, 4> Matrix<T, rows, cols>::Scale(const T& x, const T& y, const T& z) {
		Matrix<T, 4, 4> result;
		result.data[0][0] = x;
		result.data[1][1] = y;
		result.data[2][2] = z;
		return result;
	}

	template<typename T>
	Matrix<T, 4, 4> Translate(const Matrix<T, 4, 4>& m, const Vector<T, 3>& v){
		Matrix<T, 4, 4> tmp = Matrix<T, 4, 4>::Translation(v);
		return tmp * m;
	}

	template<typename T>
	Matrix<T, 4, 4> Rotate(const Matrix<T, 4, 4>& m, const Vector<T, 3>& v) {
		Matrix<T, 4, 4> tmp = Matrix<T, 4, 4>::Rotation(v);
		return tmp * m;
	}

	template<typename T>
	Matrix<T, 4, 4> Scale(const Matrix<T, 4, 4>& m, const Vector<T, 3>& v) {
		Matrix<T, 4, 4> tmp = Matrix<T, 4, 4>::Scale(v);
		return tmp * m;
	}

	// Transpose of a matrix
	template<typename T, s32 rows, s32 cols>
	Matrix<T, cols, rows> Transpose(const Matrix<T, rows, cols>& m) {
		Matrix<T, cols, rows> result;
		for (s32 i = 0; i < rows; i++) {
			for (s32 j = 0; j < cols; j++) {
				result.data[j][i] = m.data[i][j];
			}
		}

		return result;
	}

	// Matrix addition
	template<typename T, typename U, s32 rows, s32 cols>
	Matrix<T, rows, cols> operator+(const Matrix<T, rows, cols>& lhs, const Matrix<U, rows, cols>& rhs) {
		Matrix<T, rows, cols> result;
		for (s32 i = 0; i < rows; i++) {
			for (s32 j = 0; j < cols; j++) {
				result.data[i][j] = lhs.data[i][j] + static_cast<T>(rhs.data[i][j]);
			}
		}
		return result;
	}

	// Matrix substraction
	template<typename T, typename U, s32 rows, s32 cols>
	Matrix<T, rows, cols> operator-(const Matrix<T, rows, cols>& lhs, const Matrix<U, rows, cols>& rhs) {
		Matrix<T, rows, cols> result;
		for (s32 i = 0; i < rows; i++) {
			for (s32 j = 0; j < cols; j++) {
				result.data[i][j] = lhs.data[i][j] - static_cast<T>(rhs.data[i][j]);
			}
		}
		return result;
	}
	
	// Matrix multiplicated by a scalar value
	template<typename T, typename U, s32 rows, s32 cols>
	T operator*(const Matrix<T, rows, cols>& lhs, const U& rhs) {
		T result = 0;
		T s = static_cast<T>(rhs);
		for (s32 i = 0; i < rows; i++) {
			for (s32 j = 0; j < cols; j++) {
				result += lhs.data[i][j] * s;
			}
		}
		return result;
	}

	// Matrix 4x4 multiplicated by vector 3
	template<typename T, typename U>
	Vector<T, 3> operator*(const Matrix<T, 4, 4>& lhs, const Vector<U, 3>& rhs) {
		Vector<T, 3> result;
		Matrix<T, 4, 1> vm;
		vm.data[0][0] = rhs.data[0];
		vm.data[1][0] = rhs.data[1];
		vm.data[2][0] = rhs.data[2];
		vm.data[3][0] = 1;
		Matrix<T, 4, 1> r = lhs * vm;
		result.data[0] = r.data[0][0];
		result.data[1] = r.data[1][0];
		result.data[2] = r.data[2][0];
		return result;
	}

	// A matrix<m, n> multiplied by a matrix<n, p>
	template<typename T, typename U, s32 m, s32 n, s32 p>
	Matrix<T, m, p> operator*(const Matrix<T, m, n>& lhs, const Matrix<U, n, p>& rhs) {
		Matrix<T, m, p> result;
		for (s32 i = 0; i < m; i++) {
			for (s32 j = 0; j < p; j++) {
				T value = 0;
				for (s32 k = 0; k < n; k++) {
					value += lhs.data[i][k] * static_cast<T>(rhs.data[k][j]);
				}
				result.data[i][j] = value;
			}
		}
		return result;
	}

	/*
		Matrix TODO list:
		- Matrix transfomation
			- Rotate around an axis of rotation: https://learnopengl.com/#!Getting-started/Transformations
	*/

	template<typename T>
	class Quaternion {
	public:
		union {
			T data[4];
			struct { T w, x, y, z; };
			struct { T w; Vector<T, 3> xyz; };
		};
		Quaternion() : data{} { }
		Quaternion(T w, T x, T y, T z) {
			f64 a = static_cast<f64>(w) / 2;
			this->w = static_cast<T>(cos(a));
			this->x = x*sin(a);
			this->y = y*sin(a);
			this->z = z*sin(a);
		}
		Quaternion(T w, Vector<T, 3> xyz){
			f64 a = static_cast<f64>(w) / 2;
			this->w = static_cast<T>(cos(a));
			this->xyz = xyz * sin(a);
		}
		template<typename U>
		operator Quaternion<U>() {
			return Quaternion<U>(static_cast<U>(w), static_cast<Vector<U,3>>(xyz));
		}
	};

	using quatf = Quaternion<f32>;
	using quat = Quaternion<f64>;

	template<typename T, typename U>
	Quaternion<T> operator*(const Quaternion<T>& lhs, const Quaternion<U>& rhs) {
		Quaternion<T> result;
		result.w = (static_cast<T>(rhs.w) * lhs.w) - vian::Dot(rhs.xyz, lhs.xyz);
		result.xyz =  lhs.xyz * static_cast<T>(rhs.w) + rhs.xyz * lhs.w + vian::Cross(lhs.xyz, rhs.xyz);
		return result;
	}

	template<typename T, typename U>
	Vector<T, 3> operator*(const Quaternion<T>& lhs, const Vector<U, 3>& rhs) {
		Vector<T, 3> result;
		result = vian::Cross(lhs.xyz, rhs);
		result = rhs + result * (2 * lhs.w) + (vian::Cross(lhs.xyz, result) * 2);
		return result;
	}

	/*
		Quaterniont TODO list:
		- slerp
		- Quaternion * Rotation Matrix
		- Quaterion -> Rotation Matrix
		- Euler Angles -> Quaternion
		- Quaternion -> Euler Angles
	*/


	namespace experimental {

		#define MAX_BYTE		0xFF
		#define RED_MASK 		0x00FF0000
		#define GREEN_MASK		0x0000FF00
		#define BLUE_MASK		0x000000FF
		#define ALPHA_MASK		0xFF000000

		#define ALPHA_OFFSET	24
		#define RED_OFFSET		16
		#define GREEN_OFFSET	8

		using color32 = Vector<f32, 4>;
		using color24 = Vector<f32, 3>;
		using color32c = Vector<u8, 4>;
		using color24c = Vector<u8, 3>;

		inline u32 PackColor32ToARGB(const color32c& color) {
			u32 hex = (color.a << ALPHA_OFFSET) | (color.r << RED_OFFSET) | (color.g << GREEN_OFFSET) | (color.b);
			return hex;
		}
		inline u32 PackColor32ToARGB(const color32 &color)
		{
			u32 result = ((u32)color.a << ALPHA_OFFSET) | ((u32)color.r << RED_OFFSET) | ((u32)color.g << GREEN_OFFSET) | ((u32)color.b);
			return result;
		}
		inline color32 UnpackARGBToColor32(const u32 &packed_color){
			color32 result = color32(
				packed_color & RED_MASK, 
				packed_color & BLUE_MASK, 
				packed_color & GREEN_MASK, 
				packed_color & ALPHA_MASK);
			result = result / MAX_BYTE;
			return result;
		}

		inline u32 PackColor24ToARGB(const color24c& color) {
			u32 hex =  (0xFF << ALPHA_OFFSET) | (color.r << RED_OFFSET) | (color.g << GREEN_OFFSET) | (color.b);
			return hex;
		}

		inline u32 PackColor24ToARGB(const color24 &color){
			u32 result = (0xFF << ALPHA_OFFSET) | ((s32)color.r << RED_OFFSET) | ((s32)color.g << GREEN_OFFSET) | ((s32)color.b);
			return result;
		}

		inline color24 UnpackARGBToColor24(const u32 &packed_color){
			color24 result = color24(
				packed_color & RED_MASK, 
				packed_color & BLUE_MASK, 
				packed_color & GREEN_MASK);
			result = result / MAX_BYTE;
			return result;
		}

		f32 LinearToRGB(f32 L){
			if( L < 0.0f)
				L = 0.0f;
			
			if( L > 1.0f)
				L = 1.0f;
			
			f32 S = L*12.92f;
			if(L > 0.0031308f){
				S = 1.055f*pow(L, 1.0f/2.4f) - 0.055f;
			}
			return S;
		}
	}



	template<typename T, s32 size>
	std::string to_string(const Vector<T, size>& v) {
		std::string result = "(";
		for (s32 i = 0; i < size; i++) {
			result += std::to_string(v.data[i]);
			if (i < size - 1) {
				result += ", ";
			}
		}
		result += ") ";
		return result;
	}

	template<typename T, s32 rows, s32 cols>
	std::string to_string(const Matrix<T, rows, cols>& m) {
		std::string result = "";
		for (s32 i = 0; i < rows; i++) {
			for (s32 j = 0; j < cols; j++) {
				result += std::to_string(m.data[i][j]) + " ";
			}
			result += "\n";
		}
		return result;
	}

	template<typename T>
	std::string to_string(const Quaternion<T>& q) {
		std::string result = "(";
		result += std::to_string(q.w) + ", ";
		result += std::to_string(q.x) + ", ";
		result += std::to_string(q.y) + ", ";
		result += std::to_string(q.z) + ", ";
		result += ")";
		return result;
	}

	template<typename T, s32 size>
	std::ostream& operator <<(std::ostream& o, const Vector<T, size>& v) {
		o << vian::to_string(v);
		return o;
	}

	template<typename T, s32 rows, s32 cols>
	std::ostream& operator <<(std::ostream& o, const Matrix<T, rows, cols>& m) {
		o << vian::to_string(m);
		return o;
	}

	template<typename T>
	std::ostream& operator <<(std::ostream& o, const Quaternion<T>& q) {
		o << vian::to_string(q);
		return o;
	}

	inline f64 DegToRad(f64 angle) {
		return M_DEG_TO_RAD * angle;
	}

	inline f64 RadToDeg(f64 angle) {
		return M_RAD_TO_DEG * angle;
	}
}