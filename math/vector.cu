#pragma once
#include <cuda_runtime.h>

struct Vector3 {
    float x, y, z;

    __host__ __device__
    Vector3() : x(0), y(0), z(0) {

    }

    __host__ __device__
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {

    }

    __host__ __device__
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__
    Vector3 operator/(float scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    __host__ __device__
    float dot(const Vector3& other) const {
        return x * other.x + y * other.y +z * other.z;
    }

    __host__ __device__
    Vector3 cross(const Vector3& other) const {
        return Vector3(y * other.z - z * other.y,
                       z * other.x - x * other.z,
                       x * other.y - y * other.x);
    }

    float norm() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__
    Vector3 normalize() const {
        float length = norm();
        if (length == 0) {
            return Vector3(0, 0, 0); // Avoid division by zero
        }
        return Vector3(x / length, y / length, z / length);
    }

    __host__ __device__
    Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }
};